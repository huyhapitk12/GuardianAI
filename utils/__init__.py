# utils/__init__.py
"""
Unified utils package
"""

# Import from common
from .common import (
    StateManager, 
    AlertInfo, 
    SpamGuard, 
    AlarmPlayer, 
    init_alarm, 
    play_alarm, 
    stop_alarm,
    Cache,
    EmbeddingCache,
    generic_cache,
    embedding_cache,
    state_manager,
    spam_guard,
    FrameOptimizer,
    MemoryOptimizer,
    PerformanceMonitor,
    performance_timer,
    memory_usage_monitor,
    frame_optimizer,
    memory_optimizer,
    performance_monitor,
    OptimizationReporter,
    create_performance_summary,
    save_system_info,
    export_metrics_csv,
    reporter,
    cached,
    thread_safe,
    retry,
    validate_types,
    singleton,
    time_async,
    profile,
    setup_optimization_environment,
    optimize_garbage_collection,
    estimate_memory_usage,
    is_memory_available,
    get_memory_info,
    force_garbage_collection,
    batch_array_operations,
    optimize_numpy_arrays,
    clear_array_memory,
    profile_function,
    measure_function_performance,
    benchmark_operations,
    format_bytes,
    format_duration,
    safe_filename,
    get_system_info,
    check_environment,
    enable_all_optimizations,
    cleanup_resources,
    graceful_shutdown,
    thread_pool,
    task_queue
)

# Import from security
from .security import (
    SecurityManager,
    security_manager
)

__all__ = [
    'StateManager', 'AlertInfo', 'SpamGuard', 'AlarmPlayer',
    'init_alarm', 'play_alarm', 'stop_alarm',
    'Cache', 'EmbeddingCache', 'generic_cache', 'embedding_cache',
    'state_manager', 'spam_guard',
    'FrameOptimizer', 'MemoryOptimizer', 'PerformanceMonitor',
    'performance_timer', 'memory_usage_monitor',
    'frame_optimizer', 'memory_optimizer', 'performance_monitor',
    'OptimizationReporter', 'create_performance_summary',
    'save_system_info', 'export_metrics_csv', 'reporter',
    'cached', 'thread_safe', 'retry', 'validate_types', 'singleton',
    'time_async', 'profile',
    'setup_optimization_environment', 'optimize_garbage_collection',
    'estimate_memory_usage', 'is_memory_available', 'get_memory_info',
    'force_garbage_collection', 'batch_array_operations',
    'optimize_numpy_arrays', 'clear_array_memory',
    'profile_function', 'measure_function_performance', 'benchmark_operations',
    'format_bytes', 'format_duration', 'safe_filename',
    'get_system_info', 'check_environment',
    'enable_all_optimizations', 'cleanup_resources', 'graceful_shutdown',
    'thread_pool', 'task_queue',
    'SecurityManager', 'security_manager'
]