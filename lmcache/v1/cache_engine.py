# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import gc
import multiprocessing
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventType
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.storage_manager = StorageManager(
            config,
            metadata,
            # self.memory_allocator,
            event_manager=self.event_manager,
            lmcache_worker=self.lmcache_worker,
        )

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        self.use_layerwise = config.use_layerwise
        self.num_layers = metadata.kv_shape[0]
        if self.use_layerwise:
            if config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> [pinned keys]
        self.lookup_pins: dict[str, list] = defaultdict(list)

        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.post_inited = False

        # Whether to force store to wait if no CPU buffer is available
        self.force_store_wait = config.extra_config and config.extra_config.get(
            "force_store_wait", False
        )

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

    def post_init(self, **kwargs) -> None:
        if "async_lookup_server" in kwargs:
            self.async_lookup_server = kwargs.pop("async_lookup_server")
            self.storage_manager.post_init(async_lookup_server=self.async_lookup_server)
        if not self.post_inited:
            logger.info("Post-initializing LMCacheEngine")
            self.gpu_connector.initialize_kvcaches_ptr(**kwargs)
            self.post_inited = True

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []

        offload_time = 0.0
        put_time = 0.0
        tot_kv_size = 0
        tot_token_num = 0
        t = time.perf_counter()

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(
            tokens,
            hashes,
            offsets,
            mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype

            # TODO (Jiayi): should be batched in the future
            memory_obj = self.storage_manager.allocate(
                kv_shape,
                kv_dtype,
                busy_loop=self.force_store_wait,
            )
            if memory_obj is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(key)
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return
        self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        offload_time += time.perf_counter() - t

        t = time.perf_counter()

        transfer_spec = kwargs.get("transfer_spec", None)
        self.storage_manager.batched_put(keys, memory_objs, transfer_spec=transfer_spec)
        put_time += time.perf_counter() - t

        tot_time = offload_time + put_time

        logger.info(
            "Stored %d out of total %d tokens. size: %.4f gb, cost %.4f ms, "
            "throughput: %.4f GB/s; offload_time: %.4f ms, put_time: %.4f ms",
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3,
            offload_time * 1000,
            put_time * 1000,
        )

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(keys_multi_layer[0]):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.force_store_wait,
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )

            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        logger.debug(f"Stored {tot_token_num} out of total {len(tokens)} tokens")
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.
            后缀为 True 是需要检索缓存的部分
            FFFFFTTTTT 形状是为了支持“部分历史已处理”的场景（如 streaming generation）

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
            这个检查在 token_database.process_tokens 中完成
        """
        tot_kv_size = 0
        t = time.perf_counter()

        # 计算“需要检索”的 token 数量（即 mask=True 的个数）
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        
        # 向 stats_monitor 注册一次检索请求，用于指标收集（如 QPS、吞吐量）
        # TODO(lyc): 看 stats_monitor 功能
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[Tuple[CacheEngineKey, MemoryObj, int, int]] = []
        if not self._is_passive(): # 判断当前 rank 是否是“被动模式”（如在多卡设置中只接收广播而不主动检索）
            if self.async_loading:
                reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
            else:
                reordered_chunks, tot_kv_size = self._process_tokens_internal(
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
        if self.save_only_first_rank: # 一种优化策略，只有 rank 0 保存/检索缓存，其他 rank 通过通信（如 NCCL broadcast）获取
            with torch.cuda.stream(self.broadcast_stream):
                self._broadcast_or_receive_memory_objs(
                    reordered_chunks,
                    ret_mask,
                )

            # if self.gpu_connector has load_stream, self.broadcast_stream is equals
            # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
            # will execute sequentially within the stream.
            # if self.gpu_connector does not have load_stream, self.broadcast_stream
            # is created by torch.cuda.Stream(), we need to synchronize broadcast
            # operation, and then process to_cpu operation.
            if not hasattr(self.gpu_connector, "load_stream"):
                self.broadcast_stream.synchronize()

        # 将缓存数据加载到 GPU
        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
            self.gpu_connector.batched_to_gpu(
                list(memory_objs), list(starts), list(ends), **kwargs
            )

        # 清理与引用计数
        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                self.storage_manager.remove(key)
            memory_obj.ref_count_down()

        # 性能统计与日志
        onload_time = time.perf_counter() - t

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.info(
            "Retrieved %d out of %d required tokens (from %d total tokens)."
            " size: %.4f gb,"
            " cost %.4f ms, throughput: %.4f GB/s;",
            retrieved_tokens,
            num_required_tokens,
            len(tokens),
            tot_kv_size / 1024**3,
            onload_time * 1000,
            tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
        )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], Optional[List[int]], None]:
        ### -> Generator[Optional[torch.Tensor], None, None]
        # 修改：yield 接收外部传入的 selected_indices
        # selected_indices 是一个 int list，表示保留第几个 chunk
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
            第 1 次迭代：从存储后端检索第 0 层的 memory objects
            第 i 次迭代：将第 i-1 层的 memory objects 移动到 GPU，并从存储后端检索第 i 层的 memory objects
            最后一次迭代：将最后一层的 memory objects 移动到 GPU，并返回最终的检索掩码 ret_mask
            
            最后一次 yield 返回 ret_mask，其他 yield 返回 None
            特别地，第一次 yield 返回缓存命中数，用于通知调用方（如 SGLang）缓存命中情况。
        """

        assert self.storage_manager is not None
        assert self.gpu_connector is not None

        # --- 1. 预处理与统计信息 (保持原样) ---
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        
        location = None
        
        # --- 2. Token 处理与 Key 生成 (保持原样) ---
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey) # 这里的CacheEngineKey实际不存K向量，只是块的索引，MemoryObj才是存的数据

            # 将单个 CacheEngineKey 拆分为 每层独立的 key 列表
            keys_multi_layer = key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            if current_location := self.storage_manager.contains(keys_multi_layer[0]):
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        # --- 3. 流水线加载逻辑 (核心修改) ---
        if keys:
            # Transpose the keys into layer major format: [num_layers, num_chunks]
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            # [新增] 定义动态 Key 提供者
            # 使用列表作为引用容器，以便在闭包中更新当前层的 keys
            current_layer_keys_ref = [None]
            
            def key_provider_gen():
                """生成器：按需提供每一层的 keys 给 layerwise_batched_get"""
                for _ in range(self.num_layers):
                    # 每次 backend 请求任务时，都会获取最新的 keys
                    yield current_layer_keys_ref[0]

            # [新增] 初始化第 0 层（默认为全量加载）
            keys0 = keys_layer_major[0]
            current_layer_keys_ref[0] = keys0

            ### # 启动 Backend 任务生成器
            ### get_generator = self.storage_manager.layerwise_batched_get(keys_layer_major)
            # 利用 layerwise_batched_get 的异步特性，它会立即消耗 generator 并提交第一个任务
            backend_task_gen = self.storage_manager.layerwise_batched_get(
                key_provider_gen(),
                location=location,
            )

            # [Pipeline Step 1]: 立即提交第 0 层的任务 (Pre-fetch Layer 0)
            current_task = next(backend_task_gen)

            # 初始化 GPU 消费者
            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )
            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer) # 启动 consumer

            to_count_down = []
            selected_indices_current = None # 第 0 层默认全选，所以是 None

            # 进入按层循环
            for layer_id in range(self.num_layers):
                # [交互]: 将控制权交还给 Runner，通知当前层准备就绪
                # 第一层返回 token 数量，后续层返回 None
                yield_val = torch.sum(ret_mask) if layer_id == 0 else None
                # Runner 计算完当前层（或进行预处理）后，通过 send() 发送下一层的预测索引 selected_indices
                selected_indices_next = yield yield_val

                # [Pipeline Step 2]: 立即提交下一层 (Layer i+1) 的任务
                # 此时利用 Runner 发来的 selected_indices_next 进行预测加载
                next_layer_id = layer_id + 1
                if next_layer_id < self.num_layers:
                    # 根据索引筛选下一层的 keys
                    if selected_indices_next is not None:
                        next_keys = [keys_layer_major[next_layer_id][i] for i in selected_indices_next]
                    else:
                        next_keys = keys_layer_major[next_layer_id] # 没传索引则全量加载
                    
                    # 更新 Key Provider 的状态
                    current_layer_keys_ref[0] = next_keys
                    
                    # 驱动 backend 生成器继续运行，提交异步任务
                    next_task = next(backend_task_gen)
                else:
                    next_task = None

                # [Pipeline Step 3]: 等待当前层 (Layer i) 加载完成
                # 这里的 current_task 是在上一轮循环（或初始阶段）提交的
                mem_objs_layer = current_task.result()

                # [Pipeline Step 4]: 数据搬运 (H2D)
                if mem_objs_layer:
                    # 根据当前层的选择索引，筛选对应的 starts 和 ends
                    # 确保稀疏数据写入 GPU 显存的正确物理位置
                    if selected_indices_current is not None:
                        cur_starts = [starts[i] for i in selected_indices_current]
                        cur_ends = [ends[i] for i in selected_indices_current]
                    else:
                        cur_starts = starts
                        cur_ends = ends

                    # 发送元组给 GPU Connector: (数据对象, 动态Starts, 动态Ends)
                    mem_obj_consumer.send((mem_objs_layer, cur_starts, cur_ends))

                    # 收集对象以待后续引用计数递减
                    to_count_down.extend(mem_objs_layer)

                # [Pipeline Step 5]: 轮转状态
                current_task = next_task
                selected_indices_current = selected_indices_next

            # 引用计数清理
            for mem_obj in to_count_down:
                mem_obj.ref_count_down()

            # 关闭 GPU Consumer，触发最后的同步等待
            try:
                next(mem_obj_consumer)
            except StopIteration:
                pass
        
        # 处理无缓存命中的情况
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        # --- 4. 收尾逻辑 (保持原样) ---
        yield None # 同步信号

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.info(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens are cached.
        """

        if tokens is not None:
            self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            self.stats_monitor.on_lookup_request(sum(offsets))

        try:
            end = 0
            prev_end = 0

            if pin:
                assert lookup_id is not None, "lookup_id is required when pin is True"

            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)

                if self.use_layerwise:
                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    found = False
                    for key_single_layer in key_all_layers:
                        if self.storage_manager.contains(
                            key_single_layer, search_range, pin
                        ):
                            found = True
                    if found:
                        if pin:
                            self.lookup_pins[lookup_id].extend(  # type: ignore
                                key_all_layers
                            )
                        prev_end = end
                        continue
                    end = prev_end
                    return prev_end
                else:
                    if self.storage_manager.contains(key, search_range, pin):
                        if pin:
                            self.lookup_pins[lookup_id].append(  # type: ignore
                                key
                            )
                        prev_end = end
                        continue

                    end = prev_end
                    return prev_end

            # all tokens where found, return the maximal end
            return end
        finally:
            self.stats_monitor.on_lookup_finished(end)
            # vllm lookup sets pin to True
            if pin:
                self.storage_manager.touch_cache()

    # FIXME
    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """

        num_tokens = self.lookup(
            tokens,
            search_range=old_position,
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        keys = self.lookup_pins[event_id]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert memory_objs is not None, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        keys = self.lookup_pins[event_id]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        keys = self.lookup_pins[event_id]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert compressed_memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_ids: list[str]) -> None:
        for lookup_id in lookup_ids:
            if lookup_id in self.lookup_pins:
                self.storage_manager.batched_unpin(self.lookup_pins[lookup_id])
                del self.lookup_pins[lookup_id]

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                self.broadcast_object_fn(num_removed, self.metadata.first_rank)
                return num_removed
            else:
                num_removed = self.broadcast_object_fn(None, self.metadata.first_rank)
                return int(num_removed)
        return self._clear(tokens, locations, request_configs)

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""

        if self.lmcache_worker is not None:
            self.lmcache_worker.close()

        self.storage_manager.close()

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> tuple[list[tuple[CacheEngineKey, MemoryObj, int, int]], int]:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: list[tuple[CacheEngineKey, MemoryObj, int, int]] = []
        future = self.event_manager.pop_event(EventType.LOADING, kwargs["req_id"])

        memory_objs = future.result()
        memory_objs = [mm for m in memory_objs for mm in m]

        # NOTE(Jiayi): here we assume the retrieved memory_objs have
        # the same order as the lookup order.
        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        for idx, (start, end, key) in enumerate(
            self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            )
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_objs[idx]
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True

        # NOTE: free the memory objects that are not hit.
        for unused_mem_obj in memory_objs[len(chunks) :]:
            unused_mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> tuple[list[tuple[CacheEngineKey, MemoryObj, int, int]], int]:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process 待处理的 token 序列（通常是模型输入的 token ID 列表）
            mask: Mask indicating valid token positions 一个布尔掩码，标记哪些 token 位置有效（例如，跳过 padding）
            ret_mask: Output mask updated with cache hit positions 输出掩码，用于回写哪些 token 区间成功从缓存/存储中加载了 KV 缓存
            **kwargs: Additional keyword arguments

        Returns:
            A tuple containing:
            - A list of tuples, each containing (CacheEngineKey, MemoryObj, start, end)
              表示成功加载的 KV 缓存块及其对应的键和位置 (内存对象, 起始位置, 结束位置)
            - An integer representing the total size of the KV caches loaded
              成功加载的 KV 缓存总大小（以字节为单位）
        """

        tot_kv_size = 0
        # location -> [(CacheEngineKey, start, end)]
        block_mapping: dict[str, list[tuple[CacheEngineKey, int, int]]] = defaultdict(
            list
        )

        reordered_chunks: list[tuple[CacheEngineKey, MemoryObj, int, int]] = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(
            # process_tokens() 负责将输入 token 序列按某种策略切分成多个区间, 并为每个区间生成一个 CacheEngineKey
            # TODO(lyc): 是否可以实现一个自定义的 TokenDatabase 以支持区分视频和文本 token
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            # 检查 key 是否存在于存储后端
            if key in self.lookup_cache:
                # TODO(Jiayi): we can reduce the number of `contains` calls
                # by checking the lookup cache first (should be updated in `lookup`)
                pass
            else:
                # NOTE: key should always be in the lookup cache once
                # we support it.
                location = self.storage_manager.contains(key)
                if location is None:
                    break

                # NOTE: Here we make the assumption that the underlying
                # storage backend support pin operation, and the memory
                # object is already pinned in the storage backend.
                ret_mask[start:end] = True # 标记该区间的 token 的 KV 缓存可用

            assert location is not None
            
            # 按存储位置分组缓存请求, 便于后续批量获取（batched_get）
            block_mapping[location].append((key, start, end))

        # 批量从存储后端获取 MemoryObj
        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )
            assert memory_objs is not None, (
                "Failed to get memory objects from storage backend"
            )

            # 组装结果并处理失败情况
            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()

        # 处理失败后截断结果
        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                tensor_to_broadcast = memory_obj.tensor.to(
                    f"cuda:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                tensor = torch.empty(
                    metadata.shape,
                    dtype=metadata.dtype,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                torch.Size(metadata.kv_shape),
                metadata.kv_dtype,
                MemoryFormat.KV_2LTD,
            )

        if config.weka_path is not None or config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: GPUConnectorInterface,
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()
