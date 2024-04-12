sudo apt-get install python3-venv
python3.9 -m venv vnv
source vnv/bin/activate
pip3 install tensorflow==2.15 
pip3 install tensorflow_probability==0.23
pip3 install ruamel.yaml
pip3 install gym==0.26.2
pip3 install crafter==1.8.3