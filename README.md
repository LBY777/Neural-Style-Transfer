NST Image Style Transformer
===============

**This is a program that can "paint" a new images by combining the content of a Content Image with the visual style of a Style Image using pre-trained CNN model, specifically VGG16.**





* * *
How to use?
====

Make sure you have installed [python](https://www.python.org/downloads/) and [pip](https://pypi.org/project/pip/).

After cloning or downloading the repository, run the following command at the repository's root for all the dependencies.

```bash
pip3 install -r requirements.txt
```

Then, you can run

```bash
python3 main.py [path to content image] [path to style image]
```

For example, 

```bash
python3 transfer.py content/geisel.jpg style/starry.jpg
```



* * *


Sample Output
====

  
  
Here is a sample output.

  <img src="/painted/example.png" width="1000"/>

* * *

  
  



* * *
