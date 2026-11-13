# The ten-mission table

`missions.csv` is the hard-coded dataset from the original notebook (`i222327_Mohammad_Rohaan_A_03_ML.ipynb`, the “Creating Dataset” cell). The same ten rows, in the same order, appear in the assignment brief at [docs/assignment/Asg-3-ML-for-Robotics-Spring-2026-NU.pdf](../docs/assignment/Asg-3-ML-for-Robotics-Spring-2026-NU.pdf) (original filename in the clone: `Asg-3-ML for Robotics-Spring 2026-NU.pdf`).

There is no larger robot log in this repository. I wrote the CSV so the command-line script can load the table without opening the notebook.

| Column | Unit | Role |
|--------|------|------|
| `Load_Weight_kg` | kg | Predictor |
| `Speed_m_per_s` | m/s | Predictor |
| `Terrain_Roughness` | 1–10 scale (as in the brief) | Predictor |
| `Task_Duration_min` | minutes | Predictor |
| `Energy_Consumption_Wh` | Wh | Target |

Ten rows, five columns, no missing cells. Every load value is unique. This is a hand-built example table: the four inputs and the energy rise together, so the columns are strongly correlated with each other as well as with the target.
