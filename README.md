# OMR Bubble Sheet Grader

> Reading, grading and scoring cosumized students answer sheets fast and accurate based on a model published by [Udayraj123](https://github.com/Udayraj123/OMRChecker)

## Built With

- Python
- OpenCV, Numpy, pandas, deepmerge, jsonschema

## Live Demo

[Live Demo Link](https://livedemo.com) soon!


## Getting Started


To get a local copy up and running follow these simple example steps.

### Prerequisites
Operating system: Linux is recommended although Windows is also supported.

### Setup

```
cd my-folder
git clone https://github.com/ZahraArshia/OMRChecker.git
```

### Install
check if python3 and pip is already installed:

```
python3 --version
python3 -m pip --version
```

Install OpenCV:

```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```
### Project Dependencies
Install pip requirements:

```
python3 -m pip install --user -r requirements.txt
```

### Usage
1. First copy and examine the sample data to know how to structure your inputs:
```
cp -r ./samples/sample1 inputs/
# Note: you may remove previous inputs (if any) with `mv inputs/* ~/.trash`
# Change the number N in sampleN to see more examples
```
2. Run

```
python3 main.py
```
Alternatively you can also use `python3 main.py -i ./samples/sample1`.


## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues/).

## Show your support

Give a ⭐️ if you like this project!

## Acknowledgments

I would like to acknowledge [Udayraj Deshmukh](https://github.com/Udayraj123) as the author of main logic, we modfied to match our own application.

## 📝 License

This project is [MIT](./MIT.md) licensed.
