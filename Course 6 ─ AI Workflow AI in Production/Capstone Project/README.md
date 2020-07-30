# IBM AI Enterprise Workflow Certification - Capstone Solution Files


Usage notes
===========

All commands are from this directory.

To test app.py
--------------

``` {.bash}
~$ streamlit run app.py
```

The app will automatically run on localhost in your browser and you will see a basic website that can
train and test a Deep Learning Model which can be used to make predictions.

To test the model directly
--------------------------

see the code at the bottom of [model.py]{.title-ref}

``` {.bash}
~$ python model.py
```

To build the docker container
-----------------------------

``` {.bash}
~$ docker build -f Dockerfile -t app:latest .
```

Check that the image is there.

``` {.bash}
~$ docker image ls
```

You may notice images that you no longer use. You may delete them with

``` {.bash}
~$ docker image rm IMAGE_ID_OR_NAME
```

And every once and a while if you want clean up you can

``` {.bash}
~$ docker system prune
```


Run the container to test that it is working
--------------------------------------------

``` {.bash}
~$ docker run -p 8501:8501 app:latest
```

Go to <http://localhost:8501//> and  you will see a basic website that can
train and test a Deep Learning Model which can be used to make predictions.
