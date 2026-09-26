
Format of Glasgow ascii files.

line 1: sampling interval (msec), P-onset, P-offset, R-onset, R-offset, T-offset (in msec), Number of samples
line 2: Patient name, sex, age, heart rate
line 3: filename
line 4: blank

line 5-4+number of samples: sample number, time (msec), flags(4), limb leads(RA,LA,LL), nodes(1-352)

The flags are 0 or 1 for sample belongs to P-T interval, QRS, Q-T interval, J-T interval.

************************************************************************
To calculate leads from 352 node data we do the following:

Calculate Mason-Likar leads whenever true limb leads are not available or whenever
Mason-Likar leads are desired.

RA = (node60 + node101)/2
LA = (node50 + node90)/2
LL = (3*node343 + 2*node344)/5


Then:

I = LA - RA
II = LL - RA
III = LL - LA

And:

aVR = -(I + II)/2
aVL = (I - III)/2
aVF = (II + III)/2

compute Central Terminal as:

CT = (RA+LA+LL)/3

Subtract CT from each of the 352 nodes to get Mason-Likar
central terminal as a reference. Do this only if you want
to use Mason-Likar reference. Otherwise Wilson's Central
Terminal is the default.

Now:

V1 = node169
V2 = node171
V3 = (node192 + node193)/2
V4 = node216
V5 = (node217 + 2*node218)/3
V6 = node219

Other interesting leads are:

V3R = (node167 + node187 + node209 + 5*node188)/8
V4R = (node208 + node209 + node187 + node228)/4
V5R = (node186 + node227 + node208)/3
V6R = node207

V7 = node220
V8 = node221
V9 = node222


Frank, XYZ and EASI leads:

A = node198
C = node194
I = (node1165 + node207)/2        Do not confuse with lead I above!
E = node190
H = node17
M = (node182 + node202 + node203 + node223)/4
F = LL

X = .610*A + .171*C - .781*I
Y = .655*F + .345*M - H
Z = .133*A + .736*M - .264*I - .374*E - .231*C

S = node84 (We had used node44 in the past)


Our three vessel monitoring leads are:

LCx = node221 - node150
LAD = node174 - node221
RCA = node342 - node129


Jim Warren.
23-Aug-2006.

Also included is a set of five pdf files for each case plotting the maps in various formats.
For case0001 the files are;

case0001_15.pdf      Plots 12 standard leads and XYZ and a derived QRS detection function (DF).
case0001_map.pdf     Plots of 123 recorded leads in the lead placement layout.
case0001_quad.pdf    Plots of QRS for 12 standard leads with standard scales multiplied by four.
case0001_single.pdf  Plots of Q-T for 12 standard leads with standard scales.
case0001_xyz.pdf     Plots of XYZ with enlarged scales.

Other cases have similar filenames.

Jim Warren.
9-Nov-2006.
