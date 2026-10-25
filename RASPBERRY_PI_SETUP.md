# Raspberry Pi setup

These steps prepare Raspberry Pi OS and create the Python environment used by
the project. Run them on the Raspberry Pi while it is connected to the
Internet.

## 1. Install system packages

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git python3-pip python3-setuptools python3-smbus python3-venv
```

Raspberry Pi OS normally supplies `picamera2` and its `libcamera` bindings as
system packages. The environment below uses `--system-site-packages` so those
camera packages remain available.

## 2. Create the virtual environment

```bash
python3 -m venv --system-site-packages ~/env
source ~/env/bin/activate
python -m pip install --upgrade pip
```

## 3. Install the Python dependencies

```bash
git clone git@github.com:thattony/iot-lab1.git ~/cs437_lab1
cd ~/cs437_lab1
python -m pip install -r requirements.txt
```

Skip the `git clone` line if `~/cs437_lab1` already exists.

`requirements.txt` installs the `robot-hat` 2.5.x and `picar-x` 2.1.x
branches, Vilib, OpenCV, and the LiteRT inference runtime. A virtual environment
does not require `sudo pip` or `--break-system-packages`.

## 4. Verify the installation

```bash
python -c "import ai_edge_litert, cv2, picamera2, picarx, robot_hat; print('imports OK')"
python -m unittest -v
```

## 5. Optional: enable the Robot HAT I2S amplifier

The audio setup is only needed for sound output; the mapping, routing, and
object-detection programs do not require it.

```bash
cd ~
git clone --depth 1 --branch 2.5.x https://github.com/sunfounder/robot-hat.git
cd robot-hat
sudo bash i2samp.sh
```

Follow the prompts and reboot when the script requests it. If `~/robot-hat`
already exists, update that checkout instead of cloning a second copy.

## Daily use

```bash
cd ~/cs437_lab1
source ~/env/bin/activate
```
