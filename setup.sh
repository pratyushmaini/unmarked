conda create -n unmarked python=3.12 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate diffusers transformers protobuf sentencepiece
pip install -r requirements.txt