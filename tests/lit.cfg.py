import os
import lit.formats

config.name = "Ive Programming Language"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.ive', '.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'Output')

# Point to your build directory
config.substitutions.append(('%ive', os.path.join(config.test_source_root, '..', 'build', 'ive')))
