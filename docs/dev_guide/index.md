# Contributor's Guide

## Why contribute?

- A function is missing that you need;
- A function you need is present but is not fast or flexible enough for your use case;
- There is something in the package (including its documentation) that you think you could improve;
- You have done something cool and would like to share it in the examples gallery;
- You want to learn how to develop open-source scientific software or a specific skill (e.g. documentation with Sphinx);
- Out of philantropy.

## How to contribute?

1. Set up your environment and install the package with developer's options ([setup](setup.md))
2. If you do not have a specific change in mind, check out the [GitHub issues](https://github.com/Huracan-project/huracanpy/issues), or contact the developers ([Stella](https://www.physics.ox.ac.uk/our-people/bourdin) / [Leo](https://leosaffin.github.io/)). If you already do, move on to the next step.
3. Identify the sources files that require modification, and go ahead with implementing what you need. Remember to commit regularly your changes.
4. Implement relevant tests ([tests](tests.md))
5. Update or create relevant documentation ([documentation](doc.md))
6. Create a pull request for your contribution [here](https://github.com/Huracan-project/huracanpy/pulls)

If you know what it means, and think it is relevant, think about implementing your function in the `hrcn` accessor as well. 

Alternatively, if you think you have something useful to contribute like a function, an example, etc. but do not want to go all the way to inserting it yourself, please create an issue with as such information as possible and the developers will see to implementing it. 

```{toctree}
---
maxdepth: 4
hidden:
---
self
setup
examples
doc
tests
changelog
```
