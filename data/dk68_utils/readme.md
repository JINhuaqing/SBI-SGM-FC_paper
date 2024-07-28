The surface files and the template for DK 68 ( I derived from files in org_fils)

I use `from nilearn import plotting` to plot the brain, ie, plotting.plot_surf_roi

1. MyBrainMesh_ICBM152_surf_sm20_XYZs_xxx.txt: Brain 3-D coordinate files, xxx-full, left or right for full, left and right brain

2. MyBrainMesh_ICBM152_surf_sm20_faces_xxxltxt: Brain faces files, xxx-full, left or right for full, left and right brain

3. ROI_order_DK68.txt: The order of ROIs

4. brain_tmp_xxx_DK68.txt: The template file corresponding to the 3-D coordinate file (SO the lengths are the same)
                           Note that index from 1 corresponds to the ROI, 0 is the middile region of medial view (no use).
                            xxx-full(1-68), left (1-34) or right(1-34) for full, left and right brain.
                            
5. nodes_coors_DK68.csv: the node file (3-D corrdinate). I am not sure about the last two cols.




