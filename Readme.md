Generate noise

Download BDD_Dir from [Google Drive](https://drive.google.com/file/d/1k4MP97YsPkDLHHUER518q7INwE2sbFgy/view?usp=drive_link) and put them as a folder in PhantomSponges

We call the starting images set as "reference images".
Note that the perturbation always starts with all 0.

1. PhantomSponges, run_attack.py
2. CustomizedPhantomSponges 
   + single (means # reference image = 1)
   + batch (means # reference image > 1, but also restricted by the GPU mem, ~10)
   + extended-batch (ToDo, means # reference image could be larger, if we load them batch by batch)

