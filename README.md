# UH_Soundscapes

Open-access infrasound-adjacent soundscapes

Annotated openly-available low-frequency audio libraries of rocket [1], explosion [2], and hypersonic [3] signatures are provided. We use a decimated version of the Environmental Sound Classification dataset [4] to provide a validated noise vocabulary. Although the data sources vary, a majority are from smartphone and publicly available recordings. The event types are described in open-access published papers [4-9], and a Python open-source repository was developed for accessing, standardizing, and processing the curated data and metadata [10-11]. This work was funded by the Department of Energy National Nuclear Security Administration under Awards DE-NA0003920 (MTV) and DE-NA0003921 (ETI).

Main files in HIGP data repository [11]:
- ASTRA.pkl
- SHAReD.pkl
- OREX_UH_800Hz.pkl
- ESC50_800Hz.pkl
- ESC50_8kHz.pkl
- ESC50_16kHz.pkl


# Installation
Use pip:
```
pip install uh-soundscapes
```


# Overview of essential data fields included in the datasets:
<table>
  <tr>
    <th></th>
    <th>Signal source</th>
<th>Signal duration</th>
    <th>Source locations included</th>
    <th>Station locations included</th>
  </tr>
  <tr>
    <th>ASTRA.pkl</th>
    <td>Rocket launches, 800Hz</td>
    <td>Various (5-10 minutes)</td>
    <td>All</td>
    <td>All</td>
  </tr>
  <tr>
    <th>SHAReD.pkl</th>
    <td>Explosions and ambient, 800Hz and 8kHz</td>
    <td>0.96 seconds</td>
    <td>Some</td>
    <td>Some</td>
  </tr>
  <tr>
    <th>OREX_UH_800Hz.pkl</th>
    <td>Hypersonic atmospheric reentry, 800Hz</td>
    <td>1.28 seconds</td>
    <td>None*</td>
    <td>All</td>
  </tr>
  <tr>
    <th>ESC50_800Hz.pkl</th>
    <td>Environmental sounds (50 classes), downsampled to 800 Hz</td>
    <td>5 seconds</td>
    <td>None</td>
    <td>None</td>
  </tr>
  <tr>
    <th>ESC50_8kHz.pkl</th>
    <td>Environmental sounds (50 classes), downsampled to 8 kHz</td>
    <td>5 seconds</td>
    <td>None</td>
    <td>None</td>
  </tr>
  <tr>
    <th>ESC50_16kHz.pkl</th>
    <td>Environmental sounds (50 classes), downsampled to 16 kHz</td>
    <td>5 seconds</td>
    <td>None</td>
    <td>None</td>
  </tr>
</table>
*Source locations are not known, but can be estimated from OSIRIS-REx trajectory data.


# References
1.	Popenhagen, S.K. (2025).  Aggregated Smartphone Timeseries of Rocket-Generated Acoustics (ASTRA). https://doi.org/10.7910/DVN/ZKIS2K (accessed on 21 August 2025).
2.	Takazawa, S.K. (2024) Smartphone High-Explosive Audio Recordings Dataset (SHAReD). Available online: https://doi.org/10.7910/DVN/ROWODP (accessed on 21 August 2025).
3.	Silber, E.A.; Bowman, D.C.; Carr, C.G.; Eisenberg, D.P.; Elbing, B.R.; Fernando, B.; Garcés, M.A.; et al. Geophysical Observations of the 2023 September 24 OSIRIS-REx Sample Return Capsule Reentry. Planet. Sci. J. 2024, 5, 213; https://doi.org/10.3847/PSJ/ad5b5e.
4.	K. J. Piczak. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM international conference on multimedia, pp. 1015-1018, ACM, 2015. Paper available in GitHub Repository https://github.com/karolpiczak/ESC-50 (accessed on 21 August 2025).
5.	Popenhagen, S.K.; Takazawa, S.K.; Garcés, M.A. Rocket Launch Detection with Smartphone Audio and Transfer Learning. Signals 2025, 6, 41; https://doi.org/10.3390/signals6030041.
6.	Popenhagen, S.K.; Garcés, M.A. Acoustic Rocket Signatures Collected by Smartphones. Signals, 2025, 6, 5; https://doi.org/10.3390/signals6010005.
7.	Takazawa, S.K.; Popenhagen, S.K.; Ocampo Giraldo, L.A.; Hix, J.D.; Thompson, S.J.; Chichester, D.L.; Zeiler, C.P.; Garcés, M.A. Explosion Detection using Smartphones: Ensemble Learning with the Smartphone High-explosive Audio Recordings Dataset and the ESC-50 Dataset. Sensors, 2024, 24(20), 6688; https://doi.org/10.3390/s24206688.
8.	Takazawa, S.K.; Popenhagen, S.K.; Ocampo Giraldo, L.A.; Cárdenas, E.S.; Hix, J.D.; Thompson, S.J.; Chichester, D.L.; Garcés, M.A. A comparison of smartphone and infrasound microphone data from a fuel air explosive and a high explosive. J. Acoust. Soc. Am. 2024, 156, 1509-1523; https://doi.org/10.1121/10.0028379.
9.	KC, R. J; Wilson, T; Fox, D; Spillman K.; Garcés M.A.; Elbing B. R. Acoustic Observations of the OSIRIS-REx Sample Return Capsule Re-Entry from Wendover Airport. Seism. Res. Lett. 2025 XX, 1–14; https://doi.org/10.1785/0220250019.
10.	UH_Soundscapes, this repository.
11.	UH_Soundscapes, aggregated data at https://www.higp.hawaii.edu/archive/isla/UH_Soundscapes/


# Troubleshooting
## ModuleNotFoundError
If you use VSCode, there may be issues with module imports.  You'll need to add the line
```
"terminal.integrated.env.OSVERSION": {"PYTHONPATH": "${workspaceFolder}"}
```
to your user.settings.json.  *OSVERSION* is one of: `windows`, `linux`, or `osx`.

You can access the file via the title bar the top of the screen: View -> Command Palette, then choosing **Open User Settings (JSON)**. 
Add the line above on a new line before the closing brace.  You'll have to add a comma to the previous last line of the file, if any.

Once the line is added, your imports should work correctly.  If you still have issues, try restarting VS Code.

Example for Windows version (replace with env.osx for Mac):
```
{
    "files.autoSave": "afterDelay",
    "git.confirmSync": false,
    "terminal.integrated.env.windows": {"PYTHONPATH": "${workspaceFolder}"}
}
```

## pip Timeouts
Some dependencies may not install due to timeouts over VPNs or firewalls. To resolve this, try running
```
pip --timeout=10000 install 'package_name'
```
or any arbitrarily large number for the timeout value.
