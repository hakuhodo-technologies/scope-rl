## Contribution Guidelines
First off, thanks for your interest to cotribute to SCOPE-RL!

We are doing our best to make this project even better. However, we recognize that there is ample room for improvement.
We need your help to make this project even better. 
Let's make the best Off-Policy Evaluation software for Reinforcement Learning together!

We prepare some conventions as follows:

- [Coding Guidelines](#coding-guidelines)
- [Tests](#tests)
- [Continuous Integration](#continuous-integration)

## Coding Guidelines

Code is formatted with [black](https://github.com/psf/black),
and coding style is checked with [flake8](http://flake8.pycqa.org).

After installing black, you can perform code formatting by the following command:

```bash
# perform formatting recursively for the files under the current dir
$ black .
```

After installing flake8, you can check the coding style by the following command:

```bash
# perform checking of the coding style
$ flake8 .
```

## Tests

We are currently working on implementing unit testing using pytest as the testing framework. We greatly appreciate any helps for adding the test codes. If you are interested in working on the test codes, please contact: hk844@cornell.edu
<!-- We employ pytest as the testing framework. You can run all the tests as follows: -->

```bash
# perform all the tests under the tests directory
$ pytest .
```

## Continuous Integration

SCOPE-RL uses Github Actions to perform continuous integration.