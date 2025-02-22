import logging

class SwapTraceLogger:
    
    # private static instance
    __instance = None
    
    # private constructor
    def __init__(self):
        self.__swap_trace = []
        self.__logger = logging.getLogger('swap_trace_logger')
        self.__logger.setLevel(logging.DEBUG)

        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler('swap_trace.log')
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.__logger.addHandler(file_handler)
        
    def __del__(self):
        logging.shutdown()
    
    # public static method to get instance
    @staticmethod
    def get_instance():
        if SwapTraceLogger.__instance is None:
            SwapTraceLogger.__instance = SwapTraceLogger()
        return SwapTraceLogger.__instance
    
    # public method to log swap
    def log_swap(self, mapping, is_gpu_to_cpu):
        cpu_array, gpu_array = mapping
        if is_gpu_to_cpu:
            self.__logger.debug(f"Swap from GPU {gpu_array} to CPU: {cpu_array}")
        else:
            self.__logger.debug(f"Swap from CPU {cpu_array} to GPU: {gpu_array}")

    def log_allocate(self, block, sequence):
        self.__logger.debug(f"Allocate block {block} for sequence {sequence}")
    
    def log_free(self, block):
        self.__logger.debug(f"Free block {block}")
        
    # public method to get swap trace
    def get_swap_trace(self):
        return self.__swap_trace
