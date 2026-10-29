Review sample: 100 random test errors of Phikon · linear probe (10% labels, seed 0).

Open each `tiles/NNN.png`, then fill `failure_type` (suggested: label_noise, ambiguous, artifact_or_blur, stain_or_color, small_or_edge_region, look_alike_class, model_error) and `notes` in `review_sample.csv`. `labellite review --sheet <this folder>/review_sample.csv` tallies the result.

The tags in this sheet are a non-expert reading of each tile (image and Grad-CAM map). No pathologist or remote-sensing expert was involved.
