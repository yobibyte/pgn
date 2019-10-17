# pgn

PGN (read as pagan) is a library to build [graph networks](https://arxiv.org/abs/1806.01261) in pytorch.
If you love tensorflow, you can find the original implementation [here](https://github.com/deepmind/graph_nets).

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Stonehenge_-_Wiltonia_sive_Comitatus_Wiltoniensis%3B_Anglice_Wilshire_%28Atlas_van_Loon%29.jpg/1920px-Stonehenge_-_Wiltonia_sive_Comitatus_Wiltoniensis%3B_Anglice_Wilshire_%28Atlas_van_Loon%29.jpg)

## Example

Run `python3 sort.py` in the `examples' folder to run a sorting toy-task. You should get the results similar to the picture below.

![sorting example](pics/pgn_sorting_output.png)

## TODOs

* [ ] Update documentation.
* [ ] Add other examples, e.g. TSP.

## Acknowledgements

* [Wendelin Boehmer](https://whirl.cs.ox.ac.uk/member/wendelin-boehmer/) for useful discussions and trying it out.
* [Matthias Fey](https://github.com/rusty1s) for his great pytorch-scatter library.

## Disclaimer

This library is highly experimental and designed for educational purposes. For me it was a way to understand Graph Networks.

`pgn` works with `python3` only.