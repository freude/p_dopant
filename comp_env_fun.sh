#!/bin/bash

WDIR=$(pwd)

# -----------------------------------------------------------

# rm $WDIR/dis_scr/mesh_sample.mesh

# ---------------------- making mesh ------------------------
if [ ! -f "$WDIR/dis_scr/mesh_sample.mesh" ]; then

    cat > make_mesh.edp << EOF
include "mesh_gen.idp"

verbosity=0;				
real [int,int] BBB=[[-6, 6],[-6, 6],[-6, 6]];     //coordinates the cube
mesh3 Th=meshgen(BBB, 0, 0, 0, 1);             // numbers are coordinates of impurities
medit(1,Th);
savemesh(Th,"$WDIR/dis_scr/mesh_sample.mesh"); // save mesh for further processing
EOF

    FreeFem++ make_mesh.edp
    # rm make_mesh.edp

fi

# -------------------- making potential ---------------------
if [ ! -f "$WDIR/dis_scr/pot3.txt" ]; then
#    matlab -nojvm -nodisplay -nosplash -r "run('$WDIR/pot_ff.m');exit;"
    python $WDIR/pot_ff.py
fi

# ---------------- computing envelope functions  -----------
FreeFem++ $WDIR/si_ham.edp 0 0 1.0 1.0 0.19 $WDIR/dis_scr&
FreeFem++ $WDIR/si_ham.edp 0 0 1.0 0.19 1.0 $WDIR/dis_scr&
FreeFem++ $WDIR/si_ham.edp 0 0 0.19 1.0 1.0 $WDIR/dis_scr&
wait

