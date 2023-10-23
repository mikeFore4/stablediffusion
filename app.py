from pydantic import BaseModel
from flask import Flask, request

from custom_txt2img import generate_image_from_prompt

app = Flask(__name__)

class InputArgConfig(BaseModel):
    prompt: str
    outdir: str = "test_outputs"
    steps: int = 50
    plms: bool = False
    dpm: bool = False
    fixed_code: bool = False
    ddim_eta: float = 0.0
    n_iter: int = 3
    H: int = 768
    W: int = 768
    C: int = 4
    f: int = 8
    n_samples: int = 3
    n_rows: int = 0
    scale: float = 9.0
    from_file: str = None
    config: str = "configs/stable-diffusion/v2-inference-v.yaml"
    ckpt: str = "checkpoints/v2-1_768-ema-pruned.ckpt"
    seed: int = 42
    precision: str = "autocast"
    repeat: int = 1
    device: str = "cuda"
    torchscript: bool = False
    ipex: bool = False
    bf16: bool = False

def validate_input(opt):
    opt = InputArgConfig(**opt)
    return opt

@app.route('/gen_img', methods=['POST'])
def gen_img():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        opt = request.json
        opt = validate_input(opt)
    else:
        return f'Content-Type {content_type} not supported'

    response = generate_image_from_prompt(opt)

    return response

if __name__=='__main__':
    app.run(debug=True)
