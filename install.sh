pip install -U --pre triton
sudo pip install pybind11[global]
curl -L "https://developer.download.nvidia.com/assets/tools/secure/nsight-compute/2022_4_0/nsight-compute-linux-2022.4.0.15-32048681.run?hOm82gi__KHhcTihJKdIWy8RBH-_2bmvJJsCOHZnmZIONX4fPUAu_df1yLTn-nTqaANomoM_g54Nbz1QVV4AAzJHrW2AL1VDXIytxZDKpNi9mIODmy1oplgvg39UHDlQlwGTnyJZ32CdXBIxNzrZkmB3w4hTynf3RvW6s7a18bPC9QoBsC0rQCSxU-FnnCBviYIrl7Ehn4w=&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9" -o ncu.run
sudo chmod +x ./ncu.run
sudo ./ncu.run --quiet
