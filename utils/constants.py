from moog import sprite as sprite_lib


TIME_INDEX = 0  # Index of time in log file step strings
META_STATE_INDEX = 4  # Index of meta-state in log file step strings
REWARD_INDEX = 1
STATE_INDEX = 5  # Index of state in log file step strings
STATE_LAYERS = ['prey', 'cue', 'eye']  # Layers in logged state
PREY_LAYER_INDEX = 0  # Index of prey layer in state log
CUE_LAYER_INDEX = 1  # Index of cue layer in state log
EYE_LAYER_INDEX = 2  # Index of eye layer in state log

ATTRIBUTES_FULL = list(sprite_lib.Sprite.FACTOR_NAMES)
ATTRIBUTES_PARTIAL = ['x', 'y', 'opacity', 'scale', 'metadata']
ATTRIBUTES_PARTIAL_INDICES = {k: i for i, k in enumerate(ATTRIBUTES_PARTIAL)}
