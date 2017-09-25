# Contributing

Thank you for contributing to this repository.

When contributing, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

## How to contribute.
The general process to contribute to an open source project can be found in detail in this great post: [https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/).

In summary, the process consists on 7 steps:

1. Fork the repository to your GitHub account and clone it to your computer.

  ```
  $ git clone https://github.com/USERNAME/FORK.git
  ```

2. Create a upstream remote (to this repository) and sync your local copy.

  ```
  $ git remote add upstream https://github.com/MuSAELab/amplitude-modulation-analysis-module.git
  ```

  At this point `origin` refers to your forked repository (in your account) and `upstream` to the repository in [https://github.com/MuSAELab](https://github.com/MuSAELab)

  ```
  $ git checkout master    
  $ git pull upstream master && git push origin master
  ```
3. Create a branch in your local copy.

  ```
  git checkout -b fixing-something
  ```

4. Perform the change, write good commit messages.

5. Push your branch to your fork in GitHub.

  ```
  git push -u origin fixing-something
  ```

6. Create a new Pull Request.

7. Feedback and to merge your work to the `upstream` repository
