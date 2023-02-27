# PamyOnlineTargetHitting

This repository contains the code for the project:

"Online, Robust and Data-Efficient Learning of Interception Policies for a Table Tennis Robot"


It includes:

- "Online Target Hitting": contains the necessary files to use the hardware setup pamy and perform the main online optimization iteration.

- "Black-Box Model Training": contains the scripts used to train the neural network for the landing point prediction task.

- "Data Analysis": contains some of the data analysis tools used.

- "Data" contains the high-level data that was gathered, in the form of condensed information for whole online optimization runs (policy, landing points, ...).

The low-level data, that includes more detailed values for each interception (position measurements, Kalman filtered state, robot state evolution, ...) 
can also be analyzed using the functions defined in the "Data Analysis" section and can be downloaded under:

https://keeper.mpdl.mpg.de/f/f90460ed6b9b47a4b31a/
