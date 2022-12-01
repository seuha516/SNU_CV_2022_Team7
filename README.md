# Get Your Perfect Selfies
SNU CV 2022, Team 7

## How to run
### Set environment (using Docker)
_**NOTE 1**: We have set Dockerfile so that the code can work in all environments, but tensorflow may not work in some OS. (ex. Mac M1)_  
_**NOTE 2**: This repository uses git LFS because of its high-capacity files. If the file below could not be cloned due to the usage limit of git LFS, it should be added directly by the following method._  
**best-cnn-model.h5** : https://drive.google.com/file/d/1kSQ55StLWqfl1_dJzWRuQKrZHpml4csS/view?usp=share_link  
**u2net.onxx** : https://github.com/danielgatis/rembg#models

```
docker build -t gyps .
docker run --rm -it -v ${PWD}:/home -w /home gyps /bin/bash
```

### Command
```
python main.py
```
```
# You can optionally add multiple options.

# image : path of input image (default: data/default_selfie.jpg)
# background : path of background image (default: None)
# result_dir : path of result directory (default: result)
# h : retouching filter strength (default: 10)
# hColor: retouching filter strength for color components (default: 10)

python main.py \
--image=data/f1.jpg \
--background=data/b1.jpg \
--result_dir=result \
--h=10 \
--hColor=10
```

### Feature test
```
python core/detecting.py ${input image path}
python core/classifying.py ${input image path}
python core/retouching.py ${input image path}
python core/remove_background.py ${input image path}
python core/filter.py ${input image path}
```
