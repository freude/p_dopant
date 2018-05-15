#!/bin/bash

HOME=$(pwd)
SRC=$HOME/p_bulk
DATA=$SRC/p_dopant_data

cd $SRC

# -----------------------------------------------------------

rm $DATA/mesh_sample.mesh
rm $DATA/pot*

# ---------------------- making mesh ------------------------
if [ ! -f "$DATA/mesh_sample.mesh" ]; then

    cat > make_mesh.edp << EOF
include "mesh_gen.idp"

verbosity=0;				
real [int,int] BBB=[[-6, 6],[-6, 6],[-6, 6]];     //coordinates the cube
mesh3 Th=meshgen(BBB, 0, 0, 0, 1);             // numbers are coordinates of impurities
//medit(1,Th);
savemesh(Th,"$DATA/mesh_sample.mesh"); // save mesh for further processing
EOF

    echo "computting mesh"
    FreeFem++ make_mesh.edp

fi

# -------------------- making potential ---------------------
if [ ! -f "$DATA/pot3.txt" ]; then
    echo "computting potential"
    python pot_ff.py
fi

# ---------------- computing envelope functions  -----------
echo "computting envelope functions"
FreeFem++ si_ham.edp 0 0 1.0 1.0 0.19 $DATA&
FreeFem++ si_ham.edp 0 0 1.0 0.19 1.0 $DATA&
FreeFem++ si_ham.edp 0 0 0.19 1.0 1.0 $DATA&
wait

