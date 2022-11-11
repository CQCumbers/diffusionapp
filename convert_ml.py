from tensorflow import keras
from stable_diffusion_tf.stable_diffusion import StableDiffusion
import coremltools as ct
import os

# coremltools/converters/mil/frontend/tensorflow/ops.py:1850
# constant_val = _np.float16(node.attr.get("constant_val", 0.0))
from coremltools.converters.mil.frontend.tensorflow2.load import TF2Loader
from tensorflow.python.framework.convert_to_constants import (
    _replace_variables_by_constants, _FunctionConverterDataInEager
)

keras.mixed_precision.set_global_policy("float16")
generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False)

def _graph_def_from_concrete_fn(self, cfs):
    converter_data = _FunctionConverterDataInEager(
        func=cfs[0], lower_control_flow=False, aggressive_inlining=False)
    graph_def, _ = _replace_variables_by_constants(converter_data=converter_data)
    return graph_def

TF2Loader._graph_def_from_concrete_fn = _graph_def_from_concrete_fn


def convert_model(name, model):
    print(f'Converting {name}')
    if not os.path.exists(f'{name}_saved'):
        model.save(f'{name}_saved', save_format='tf', include_optimizer=False)
    ml = ct.convert(f'{name}_saved',
        source='tensorflow',
        convert_to='mlprogram')
        #compute_precision=ct.precision.FLOAT16)
    ml.save(f'{name}.mlpackage')

if __name__ == '__main__':
    convert_model('diffuser', generator.diffusion_model)
    convert_model('txt_encoder', generator.text_encoder)
    convert_model('img_encoder', generator.encoder)
    convert_model('decoder', generator.decoder)


