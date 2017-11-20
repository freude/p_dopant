#!/bin/bash

HOME=$(pwd)
SRC=$HOME/p_bulk
DATA=$HOME/p_dopant_data

# -----------------------------------------------------------

# rm $WDIR/dis_scr/mesh_sample.mesh

# ---------------------- making mesh ------------------------
if [ ! -f "$DATA/mesh_sample.mesh" ]; then

    cat > make_mesh.edp << EOF
include "mesh_gen.idp"

verbosity=0;				
real [int,int] BBB=[[-6, 6],[-6, 6],[-6, 6]];     //coordinates the cube
mesh3 Th=meshgen(BBB, 0, 0, 0, 1);             // numbers are coordinates of impurities
medit(1,Th);
savemesh(Th,"$DATA/mesh_sample.mesh"); // save mesh for further processing
EOF

    FreeFem++ make_mesh.edp

fi

# -------------------- making potential ---------------------
if [ ! -f "$DATA/pot3.txt" ]; then
    python $SRC/pot_ff.py
fi

# ---------------- computing envelope functions  -----------
FreeFem++ $SRC/si_ham.edp 0 0 1.0 1.0 0.19 $DATA&
FreeFem++ $SRC/si_ham.edp 0 0 1.0 0.19 1.0 $DATA&
FreeFem++ $SRC/si_ham.edp 0 0 0.19 1.0 1.0 $DATA&
wait

