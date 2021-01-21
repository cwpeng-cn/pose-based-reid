pip install pyyaml==5.2
pip install scipy==1.1.0
pip install torch==1.2.0 torchvision==0.4.0
pip install pillow==6.2.2

python -m pip install cython
sudo apt-get install libyaml-dev

cd Alphapose
python setup.py build develop

pip install -U -q PyDrive