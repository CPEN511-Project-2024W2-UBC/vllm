
# singuleton class to log swap trace
import logging


class SwapTraceLogger():
    
    # private static instance
    __instance = None
    __logger = None
    
    # private constructor
    def __init__(self):
        self.__swap_trace = []
        self.__logger = logging.getLogger('swap_trace_logger')
        
    # public static method to get instance
    @staticmethod
    def get_instance():
        if SwapTraceLogger.__instance == None:
            SwapTraceLogger.__instance = SwapTraceLogger()
        return SwapTraceLogger.__instance
    
    # public method to log swap
    def log_swap(self, cpu_block_id, gpu_block_id, mapping):
        self.__swap_trace.append((cpu_block_id, gpu_block_id))
        self.__logger.info('swap: cpu_block_id: %d, gpu_block_id: %d, mapping %d -> %d' % (cpu_block_id, gpu_block_id, mapping[0], mapping[1]))
        
    # public method to get swap trace
    def get_swap_trace(self):
        return self.__swap_trace
    
    
            
    