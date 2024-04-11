# Translation-Project
## Project structure
The project is called `TranslationWebsite`.

## Installation instructions
To install the software and use it in your local development environment, you must first set up and activate a local development environment.  From the root of the project:

```
$ virtualenv venv
$ source venv/bin/activate
```

Install all required packages:

```
$ pip3 install -r requirements.txt
```

Migrate the database:

```
$ python3 manage.py makemigrations
```

```
$ python3 manage.py migrate
```

run the server with:

```
$ python3 manage.py  runserver
```
Run all tests with:
```
$ python3 manage.py test
```


## Sources
The packages used by this application are specified in `requirements.txt`

*Declare are other sources here.*
