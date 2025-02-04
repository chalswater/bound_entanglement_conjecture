# Proof of the conjecture for the paper: 
## Bound entangled states are useful in prepare-and-measure scenarios

Here one can find the codes in python to prove the conjecture in equation (7) of the paper titled "_Bound entangled states are useful in prepare-and-measure scenarios_".

The files are the following:

1. **sdp_relaxation.ipynb**: A _jupyter notebook_ file with the SDP relaxation to compute bounds of _Rd_
2. **raw_sdp_relaxation.py**: The same SDP relaxation but in a _.py_ file.
3. **main.py**: The same SDP relaxation but constructed using the package: _MoMPy_ to reduce the number of variables. Use this one to compute high-dimensional cases, although it may take a long time before the simplification is completed.
4. **functions.py**: Funnctions required to run the **main.py** file.
5. **conjecture_sym_blkdiag_primal.py**: Primal SDP to compute bounds on the conjecture form the main text.
6. **conjecture_sym_blkdiag_dual.py**: Dual SDP to compute bounds on the conjecture form the main text. Proof of the conjecture.

 
