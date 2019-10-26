; Manually optimized with hdfsee from another set of (fatter) sucrose rings (25.10.2019)
; Manually optimized with hdfsee from sucrose rings (25.10.2019)
; Manually optimized-ish with hdfsee from silver behenate rings (23.05.2019)
; Manually optimized with hdfsee
; Manually optimized with hdfsee
; Manually optimized with hdfsee
; CrystFEL geometry file spit out of cppxfel from 120 crystals.
; Changed photon energy from 9370 to 9337
; CrystFEL geometry file spit out of cppxfel from 84 crystals.
; CrystFEL geometry file spit out of cppxfel from 28 crystals.
; Manually optimized with hdfsee
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Camera length from LiTiO calibration
; Manually optimized with hdfsee
; Now all distances between panels is 5.8mm (29 ixels)
; OY: ACHTUNG! Orientation of the 2 halves of the detector might be wrong!
; A bit changed by OY: now 128 panels, rigid groups, bad lines to mask double pixels.
; Fixed errors noticed by Oleksandr
; Beginning of an AGIPD geometry construction by Helen Ginn on Thursday before beam time.
; Global panel positions not guesses
; Local positioning largely - but not entirely - guesses
; fast and slow scan directions have the potential to be accurate.

adu_per_eV = 0.0001  ; no idea
clen = 0.1244
photon_energy = 9337
res = 5000 ; 200 um pixels

dim0 = %
dim1 = ss
dim2 = fs
data = /data/data
;data = /entry_1/instrument_1/detector_1/data
;data = /data

rigid_group_q0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7,p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7,p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p2a7,p3a0,p3a1,p3a2,p3a3,p3a4,p3a5,p3a6,p3a7
rigid_group_q1 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7,p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7,p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7,p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_q2 = p8a0,p8a1,p8a2,p8a3,p8a4,p8a5,p8a6,p8a7,p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a6,p9a7,p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7,p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_q3 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7,p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7,p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p14a7,p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_p0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7
rigid_group_p1 = p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7
rigid_group_p2 = p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p2a7
rigid_group_p3 = p3a0,p3a1,p3a2,p3a3,p3a4,p3a5,p3a6,p3a7
rigid_group_p4 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7
rigid_group_p5 = p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7
rigid_group_p6 = p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7
rigid_group_p7 = p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_p8 = p8a0,p8a1,p8a2,p8a3,p8a4,p8a5,p8a6,p8a7
rigid_group_p9 = p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a6,p9a7
rigid_group_p10 = p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7
rigid_group_p11 = p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_p12 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7
rigid_group_p13 = p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7
rigid_group_p14 = p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p14a7
rigid_group_p15 = p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15

p0a0/min_fs = 0
p0a0/min_ss = 0
p0a0/max_fs = 127
p0a0/max_ss = 63
p0a0/res = 5000.0000000000
p0a0/coffset = -0.0000187287
p0a0/fs = +0.001918x -0.999997y
p0a0/ss = +0.999996x +0.001915y
p0a0/corner_x = -537.012
p0a0/corner_y = 624.132

p0a1/min_fs = 0
p0a1/min_ss = 64
p0a1/max_fs = 127
p0a1/max_ss = 127
p0a1/res = 5000.0000000000
p0a1/coffset = 0.0000055419
p0a1/fs = +0.001918x -0.999997y
p0a1/ss = +0.999996x +0.001915y
p0a1/corner_x = -471.477
p0a1/corner_y = 624.257

p0a2/min_fs = 0
p0a2/min_ss = 128
p0a2/max_fs = 127
p0a2/max_ss = 191
p0a2/res = 5000.0000000000
p0a2/coffset = 0.0000298121
p0a2/fs = +0.001918x -0.999997y
p0a2/ss = +0.999996x +0.001915y
p0a2/corner_x = -405.944
p0a2/corner_y = 624.383

p0a3/min_fs = 0
p0a3/min_ss = 192
p0a3/max_fs = 127
p0a3/max_ss = 255
p0a3/res = 5000.0000000000
p0a3/coffset = 0.0000540827
p0a3/fs = +0.001918x -0.999997y
p0a3/ss = +0.999996x +0.001915y
p0a3/corner_x = -340.409
p0a3/corner_y = 624.508

p0a4/min_fs = 0
p0a4/min_ss = 256
p0a4/max_fs = 127
p0a4/max_ss = 319
p0a4/res = 5000.0000000000
p0a4/coffset = 0.0000783533
p0a4/fs = +0.001918x -0.999997y
p0a4/ss = +0.999996x +0.001915y
p0a4/corner_x = -274.874
p0a4/corner_y = 624.634

p0a5/min_fs = 0
p0a5/min_ss = 320
p0a5/max_fs = 127
p0a5/max_ss = 383
p0a5/res = 5000.0000000000
p0a5/coffset = 0.0001026238
p0a5/fs = +0.001918x -0.999997y
p0a5/ss = +0.999996x +0.001915y
p0a5/corner_x = -209.339
p0a5/corner_y = 624.759

p0a6/min_fs = 0
p0a6/min_ss = 384
p0a6/max_fs = 127
p0a6/max_ss = 447
p0a6/res = 5000.0000000000
p0a6/coffset = 0.0001268940
p0a6/fs = +0.001918x -0.999997y
p0a6/ss = +0.999996x +0.001915y
p0a6/corner_x = -143.806
p0a6/corner_y = 624.885

p0a7/min_fs = 0
p0a7/min_ss = 448
p0a7/max_fs = 127
p0a7/max_ss = 511
p0a7/res = 5000.0000000000
p0a7/coffset = 0.0001511648
p0a7/fs = +0.001918x -0.999997y
p0a7/ss = +0.999996x +0.001915y
p0a7/corner_x = -78.2702
p0a7/corner_y = 625.01

p1a0/min_fs = 0
p1a0/min_ss = 512
p1a0/max_fs = 127
p1a0/max_ss = 575
p1a0/res = 5000.0000000000
p1a0/coffset = 0.0000247401
p1a0/fs = +0.000841x -0.999997y
p1a0/ss = +0.999999x +0.000838y
p1a0/corner_x = -536.143
p1a0/corner_y = 468.414

p1a1/min_fs = 0
p1a1/min_ss = 576
p1a1/max_fs = 127
p1a1/max_ss = 639
p1a1/res = 5000.0000000000
p1a1/coffset = 0.0000415274
p1a1/fs = +0.000841x -0.999997y
p1a1/ss = +0.999999x +0.000838y
p1a1/corner_x = -470.609
p1a1/corner_y = 468.469

p1a2/min_fs = 0
p1a2/min_ss = 640
p1a2/max_fs = 127
p1a2/max_ss = 703
p1a2/res = 5000.0000000000
p1a2/coffset = 0.0000583149
p1a2/fs = +0.000841x -0.999997y
p1a2/ss = +0.999999x +0.000838y
p1a2/corner_x = -405.074
p1a2/corner_y = 468.523

p1a3/min_fs = 0
p1a3/min_ss = 704
p1a3/max_fs = 127
p1a3/max_ss = 767
p1a3/res = 5000.0000000000
p1a3/coffset = 0.0000751024
p1a3/fs = +0.000841x -0.999997y
p1a3/ss = +0.999999x +0.000838y
p1a3/corner_x = -339.539
p1a3/corner_y = 468.578

p1a4/min_fs = 0
p1a4/min_ss = 768
p1a4/max_fs = 127
p1a4/max_ss = 831
p1a4/res = 5000.0000000000
p1a4/coffset = 0.0000918899
p1a4/fs = +0.000841x -0.999997y
p1a4/ss = +0.999999x +0.000838y
p1a4/corner_x = -274.004
p1a4/corner_y = 468.633

p1a5/min_fs = 0
p1a5/min_ss = 832
p1a5/max_fs = 127
p1a5/max_ss = 895
p1a5/res = 5000.0000000000
p1a5/coffset = 0.0001086772
p1a5/fs = +0.000841x -0.999997y
p1a5/ss = +0.999999x +0.000838y
p1a5/corner_x = -208.47
p1a5/corner_y = 468.688

p1a6/min_fs = 0
p1a6/min_ss = 896
p1a6/max_fs = 127
p1a6/max_ss = 959
p1a6/res = 5000.0000000000
p1a6/coffset = 0.0001254647
p1a6/fs = +0.000841x -0.999997y
p1a6/ss = +0.999999x +0.000838y
p1a6/corner_x = -142.935
p1a6/corner_y = 468.743

p1a7/min_fs = 0
p1a7/min_ss = 960
p1a7/max_fs = 127
p1a7/max_ss = 1023
p1a7/res = 5000.0000000000
p1a7/coffset = 0.0001422523
p1a7/fs = +0.000841x -0.999997y
p1a7/ss = +0.999999x +0.000838y
p1a7/corner_x = -77.4002
p1a7/corner_y = 468.798

p2a0/min_fs = 0
p2a0/min_ss = 1024
p2a0/max_fs = 127
p2a0/max_ss = 1087
p2a0/res = 5000.0000000000
p2a0/coffset = -0.0002587023
p2a0/fs = +0.000489x -0.999998y
p2a0/ss = +1.000000x +0.000488y
p2a0/corner_x = -534.987
p2a0/corner_y = 312.275

p2a1/min_fs = 0
p2a1/min_ss = 1088
p2a1/max_fs = 127
p2a1/max_ss = 1151
p2a1/res = 5000.0000000000
p2a1/coffset = -0.0002539673
p2a1/fs = +0.000489x -0.999998y
p2a1/ss = +1.000000x +0.000488y
p2a1/corner_x = -469.453
p2a1/corner_y = 312.307

p2a2/min_fs = 0
p2a2/min_ss = 1152
p2a2/max_fs = 127
p2a2/max_ss = 1215
p2a2/res = 5000.0000000000
p2a2/coffset = -0.0002492323
p2a2/fs = +0.000489x -0.999998y
p2a2/ss = +1.000000x +0.000488y
p2a2/corner_x = -403.918
p2a2/corner_y = 312.339

p2a3/min_fs = 0
p2a3/min_ss = 1216
p2a3/max_fs = 127
p2a3/max_ss = 1279
p2a3/res = 5000.0000000000
p2a3/coffset = -0.0002444972
p2a3/fs = +0.000489x -0.999998y
p2a3/ss = +1.000000x +0.000488y
p2a3/corner_x = -338.384
p2a3/corner_y = 312.371

p2a4/min_fs = 0
p2a4/min_ss = 1280
p2a4/max_fs = 127
p2a4/max_ss = 1343
p2a4/res = 5000.0000000000
p2a4/coffset = -0.0002397621
p2a4/fs = +0.000489x -0.999998y
p2a4/ss = +1.000000x +0.000488y
p2a4/corner_x = -272.849
p2a4/corner_y = 312.403

p2a5/min_fs = 0
p2a5/min_ss = 1344
p2a5/max_fs = 127
p2a5/max_ss = 1407
p2a5/res = 5000.0000000000
p2a5/coffset = -0.0002350272
p2a5/fs = +0.000489x -0.999998y
p2a5/ss = +1.000000x +0.000488y
p2a5/corner_x = -207.315
p2a5/corner_y = 312.435

p2a6/min_fs = 0
p2a6/min_ss = 1408
p2a6/max_fs = 127
p2a6/max_ss = 1471
p2a6/res = 5000.0000000000
p2a6/coffset = -0.0002302921
p2a6/fs = +0.000489x -0.999998y
p2a6/ss = +1.000000x +0.000488y
p2a6/corner_x = -141.78
p2a6/corner_y = 312.467

p2a7/min_fs = 0
p2a7/min_ss = 1472
p2a7/max_fs = 127
p2a7/max_ss = 1535
p2a7/res = 5000.0000000000
p2a7/coffset = -0.0002255570
p2a7/fs = +0.000489x -0.999998y
p2a7/ss = +1.000000x +0.000488y
p2a7/corner_x = -76.2444
p2a7/corner_y = 312.499

p3a0/min_fs = 0
p3a0/min_ss = 1536
p3a0/max_fs = 127
p3a0/max_ss = 1599
p3a0/res = 5000.0000000000
p3a0/coffset = -0.0003009005
p3a0/fs = +0.001241x -0.999999y
p3a0/ss = +0.999999x +0.001241y
p3a0/corner_x = -534.535
p3a0/corner_y = 155.608

p3a1/min_fs = 0
p3a1/min_ss = 1600
p3a1/max_fs = 127
p3a1/max_ss = 1663
p3a1/res = 5000.0000000000
p3a1/coffset = -0.0003135559
p3a1/fs = +0.001241x -0.999999y
p3a1/ss = +0.999999x +0.001241y
p3a1/corner_x = -469.001
p3a1/corner_y = 155.689

p3a2/min_fs = 0
p3a2/min_ss = 1664
p3a2/max_fs = 127
p3a2/max_ss = 1727
p3a2/res = 5000.0000000000
p3a2/coffset = -0.0003262115
p3a2/fs = +0.001241x -0.999999y
p3a2/ss = +0.999999x +0.001241y
p3a2/corner_x = -403.466
p3a2/corner_y = 155.77

p3a3/min_fs = 0
p3a3/min_ss = 1728
p3a3/max_fs = 127
p3a3/max_ss = 1791
p3a3/res = 5000.0000000000
p3a3/coffset = -0.0003388670
p3a3/fs = +0.001241x -0.999999y
p3a3/ss = +0.999999x +0.001241y
p3a3/corner_x = -337.931
p3a3/corner_y = 155.852

p3a4/min_fs = 0
p3a4/min_ss = 1792
p3a4/max_fs = 127
p3a4/max_ss = 1855
p3a4/res = 5000.0000000000
p3a4/coffset = -0.0003515226
p3a4/fs = +0.001241x -0.999999y
p3a4/ss = +0.999999x +0.001241y
p3a4/corner_x = -272.396
p3a4/corner_y = 155.933

p3a5/min_fs = 0
p3a5/min_ss = 1856
p3a5/max_fs = 127
p3a5/max_ss = 1919
p3a5/res = 5000.0000000000
p3a5/coffset = -0.0003641780
p3a5/fs = +0.001241x -0.999999y
p3a5/ss = +0.999999x +0.001241y
p3a5/corner_x = -206.862
p3a5/corner_y = 156.014

p3a6/min_fs = 0
p3a6/min_ss = 1920
p3a6/max_fs = 127
p3a6/max_ss = 1983
p3a6/res = 5000.0000000000
p3a6/coffset = -0.0003768336
p3a6/fs = +0.001241x -0.999999y
p3a6/ss = +0.999999x +0.001241y
p3a6/corner_x = -141.327
p3a6/corner_y = 156.096

p3a7/min_fs = 0
p3a7/min_ss = 1984
p3a7/max_fs = 127
p3a7/max_ss = 2047
p3a7/res = 5000.0000000000
p3a7/coffset = -0.0003894891
p3a7/fs = +0.001241x -0.999999y
p3a7/ss = +0.999999x +0.001241y
p3a7/corner_x = -75.7929
p3a7/corner_y = 156.177

p4a0/min_fs = 0
p4a0/min_ss = 2048
p4a0/max_fs = 127
p4a0/max_ss = 2111
p4a0/res = 5000.0000000000
p4a0/coffset = -0.0004987626
p4a0/fs = -0.000185x -1.000000y
p4a0/ss = +1.000000x -0.000185y
p4a0/corner_x = -544.307
p4a0/corner_y = 3.65518

p4a1/min_fs = 0
p4a1/min_ss = 2112
p4a1/max_fs = 127
p4a1/max_ss = 2175
p4a1/res = 5000.0000000000
p4a1/coffset = -0.0004997256
p4a1/fs = -0.000185x -1.000000y
p4a1/ss = +1.000000x -0.000185y
p4a1/corner_x = -478.772
p4a1/corner_y = 3.64304

p4a2/min_fs = 0
p4a2/min_ss = 2176
p4a2/max_fs = 127
p4a2/max_ss = 2239
p4a2/res = 5000.0000000000
p4a2/coffset = -0.0005006885
p4a2/fs = -0.000185x -1.000000y
p4a2/ss = +1.000000x -0.000185y
p4a2/corner_x = -413.237
p4a2/corner_y = 3.6309

p4a3/min_fs = 0
p4a3/min_ss = 2240
p4a3/max_fs = 127
p4a3/max_ss = 2303
p4a3/res = 5000.0000000000
p4a3/coffset = -0.0005016515
p4a3/fs = -0.000185x -1.000000y
p4a3/ss = +1.000000x -0.000185y
p4a3/corner_x = -347.702
p4a3/corner_y = 3.61876

p4a4/min_fs = 0
p4a4/min_ss = 2304
p4a4/max_fs = 127
p4a4/max_ss = 2367
p4a4/res = 5000.0000000000
p4a4/coffset = -0.0005026144
p4a4/fs = -0.000185x -1.000000y
p4a4/ss = +1.000000x -0.000185y
p4a4/corner_x = -282.168
p4a4/corner_y = 3.60662

p4a5/min_fs = 0
p4a5/min_ss = 2368
p4a5/max_fs = 127
p4a5/max_ss = 2431
p4a5/res = 5000.0000000000
p4a5/coffset = -0.0005035773
p4a5/fs = -0.000185x -1.000000y
p4a5/ss = +1.000000x -0.000185y
p4a5/corner_x = -216.633
p4a5/corner_y = 3.59448

p4a6/min_fs = 0
p4a6/min_ss = 2432
p4a6/max_fs = 127
p4a6/max_ss = 2495
p4a6/res = 5000.0000000000
p4a6/coffset = -0.0005045403
p4a6/fs = -0.000185x -1.000000y
p4a6/ss = +1.000000x -0.000185y
p4a6/corner_x = -151.098
p4a6/corner_y = 3.58234

p4a7/min_fs = 0
p4a7/min_ss = 2496
p4a7/max_fs = 127
p4a7/max_ss = 2559
p4a7/res = 5000.0000000000
p4a7/coffset = -0.0005055032
p4a7/fs = -0.000185x -1.000000y
p4a7/ss = +1.000000x -0.000185y
p4a7/corner_x = -85.5635
p4a7/corner_y = 3.5702

p5a0/min_fs = 0
p5a0/min_ss = 2560
p5a0/max_fs = 127
p5a0/max_ss = 2623
p5a0/res = 5000.0000000000
p5a0/coffset = -0.0000100485
p5a0/fs = +0.000048x -1.000000y
p5a0/ss = +1.000000x +0.000048y
p5a0/corner_x = -544.797
p5a0/corner_y = -153.951

p5a1/min_fs = 0
p5a1/min_ss = 2624
p5a1/max_fs = 127
p5a1/max_ss = 2687
p5a1/res = 5000.0000000000
p5a1/coffset = -0.0000181948
p5a1/fs = +0.000048x -1.000000y
p5a1/ss = +1.000000x +0.000048y
p5a1/corner_x = -479.262
p5a1/corner_y = -153.948

p5a2/min_fs = 0
p5a2/min_ss = 2688
p5a2/max_fs = 127
p5a2/max_ss = 2751
p5a2/res = 5000.0000000000
p5a2/coffset = -0.0000263412
p5a2/fs = +0.000048x -1.000000y
p5a2/ss = +1.000000x +0.000048y
p5a2/corner_x = -413.727
p5a2/corner_y = -153.944

p5a3/min_fs = 0
p5a3/min_ss = 2752
p5a3/max_fs = 127
p5a3/max_ss = 2815
p5a3/res = 5000.0000000000
p5a3/coffset = -0.0000344876
p5a3/fs = +0.000048x -1.000000y
p5a3/ss = +1.000000x +0.000048y
p5a3/corner_x = -348.192
p5a3/corner_y = -153.941

p5a4/min_fs = 0
p5a4/min_ss = 2816
p5a4/max_fs = 127
p5a4/max_ss = 2879
p5a4/res = 5000.0000000000
p5a4/coffset = -0.0000426338
p5a4/fs = +0.000048x -1.000000y
p5a4/ss = +1.000000x +0.000048y
p5a4/corner_x = -282.658
p5a4/corner_y = -153.938

p5a5/min_fs = 0
p5a5/min_ss = 2880
p5a5/max_fs = 127
p5a5/max_ss = 2943
p5a5/res = 5000.0000000000
p5a5/coffset = -0.0000507801
p5a5/fs = +0.000048x -1.000000y
p5a5/ss = +1.000000x +0.000048y
p5a5/corner_x = -217.123
p5a5/corner_y = -153.935

p5a6/min_fs = 0
p5a6/min_ss = 2944
p5a6/max_fs = 127
p5a6/max_ss = 3007
p5a6/res = 5000.0000000000
p5a6/coffset = -0.0000589265
p5a6/fs = +0.000048x -1.000000y
p5a6/ss = +1.000000x +0.000048y
p5a6/corner_x = -151.588
p5a6/corner_y = -153.932

p5a7/min_fs = 0
p5a7/min_ss = 3008
p5a7/max_fs = 127
p5a7/max_ss = 3071
p5a7/res = 5000.0000000000
p5a7/coffset = -0.0000670728
p5a7/fs = +0.000048x -1.000000y
p5a7/ss = +1.000000x +0.000048y
p5a7/corner_x = -86.0535
p5a7/corner_y = -153.929

p6a0/min_fs = 0
p6a0/min_ss = 3072
p6a0/max_fs = 127
p6a0/max_ss = 3135
p6a0/res = 5000.0000000000
p6a0/coffset = 0.0004976381
p6a0/fs = +0.000039x -1.000000y
p6a0/ss = +0.999999x +0.000038y
p6a0/corner_x = -545.296
p6a0/corner_y = -311.479

p6a1/min_fs = 0
p6a1/min_ss = 3136
p6a1/max_fs = 127
p6a1/max_ss = 3199
p6a1/res = 5000.0000000000
p6a1/coffset = 0.0004805617
p6a1/fs = +0.000039x -1.000000y
p6a1/ss = +0.999999x +0.000038y
p6a1/corner_x = -479.761
p6a1/corner_y = -311.477

p6a2/min_fs = 0
p6a2/min_ss = 3200
p6a2/max_fs = 127
p6a2/max_ss = 3263
p6a2/res = 5000.0000000000
p6a2/coffset = 0.0004634855
p6a2/fs = +0.000039x -1.000000y
p6a2/ss = +0.999999x +0.000038y
p6a2/corner_x = -414.227
p6a2/corner_y = -311.474

p6a3/min_fs = 0
p6a3/min_ss = 3264
p6a3/max_fs = 127
p6a3/max_ss = 3327
p6a3/res = 5000.0000000000
p6a3/coffset = 0.0004464091
p6a3/fs = +0.000039x -1.000000y
p6a3/ss = +0.999999x +0.000038y
p6a3/corner_x = -348.692
p6a3/corner_y = -311.472

p6a4/min_fs = 0
p6a4/min_ss = 3328
p6a4/max_fs = 127
p6a4/max_ss = 3391
p6a4/res = 5000.0000000000
p6a4/coffset = 0.0004293327
p6a4/fs = +0.000039x -1.000000y
p6a4/ss = +0.999999x +0.000038y
p6a4/corner_x = -283.157
p6a4/corner_y = -311.469

p6a5/min_fs = 0
p6a5/min_ss = 3392
p6a5/max_fs = 127
p6a5/max_ss = 3455
p6a5/res = 5000.0000000000
p6a5/coffset = 0.0004122563
p6a5/fs = +0.000039x -1.000000y
p6a5/ss = +0.999999x +0.000038y
p6a5/corner_x = -217.622
p6a5/corner_y = -311.467

p6a6/min_fs = 0
p6a6/min_ss = 3456
p6a6/max_fs = 127
p6a6/max_ss = 3519
p6a6/res = 5000.0000000000
p6a6/coffset = 0.0003951802
p6a6/fs = +0.000039x -1.000000y
p6a6/ss = +0.999999x +0.000038y
p6a6/corner_x = -152.088
p6a6/corner_y = -311.464

p6a7/min_fs = 0
p6a7/min_ss = 3520
p6a7/max_fs = 127
p6a7/max_ss = 3583
p6a7/res = 5000.0000000000
p6a7/coffset = 0.0003781037
p6a7/fs = +0.000039x -1.000000y
p6a7/ss = +0.999999x +0.000038y
p6a7/corner_x = -86.5534
p6a7/corner_y = -311.462

p7a0/min_fs = 0
p7a0/min_ss = 3584
p7a0/max_fs = 127
p7a0/max_ss = 3647
p7a0/res = 5000.0000000000
p7a0/coffset = 0.0000550633
p7a0/fs = -0.000100x -1.000000y
p7a0/ss = +1.000000x -0.000100y
p7a0/corner_x = -544.94
p7a0/corner_y = -464.872

p7a1/min_fs = 0
p7a1/min_ss = 3648
p7a1/max_fs = 127
p7a1/max_ss = 3711
p7a1/res = 5000.0000000000
p7a1/coffset = 0.0000517840
p7a1/fs = -0.000100x -1.000000y
p7a1/ss = +1.000000x -0.000100y
p7a1/corner_x = -479.406
p7a1/corner_y = -464.878

p7a2/min_fs = 0
p7a2/min_ss = 3712
p7a2/max_fs = 127
p7a2/max_ss = 3775
p7a2/res = 5000.0000000000
p7a2/coffset = 0.0000485046
p7a2/fs = -0.000100x -1.000000y
p7a2/ss = +1.000000x -0.000100y
p7a2/corner_x = -413.871
p7a2/corner_y = -464.885

p7a3/min_fs = 0
p7a3/min_ss = 3776
p7a3/max_fs = 127
p7a3/max_ss = 3839
p7a3/res = 5000.0000000000
p7a3/coffset = 0.0000452253
p7a3/fs = -0.000100x -1.000000y
p7a3/ss = +1.000000x -0.000100y
p7a3/corner_x = -348.336
p7a3/corner_y = -464.891

p7a4/min_fs = 0
p7a4/min_ss = 3840
p7a4/max_fs = 127
p7a4/max_ss = 3903
p7a4/res = 5000.0000000000
p7a4/coffset = 0.0000419459
p7a4/fs = -0.000100x -1.000000y
p7a4/ss = +1.000000x -0.000100y
p7a4/corner_x = -282.801
p7a4/corner_y = -464.898

p7a5/min_fs = 0
p7a5/min_ss = 3904
p7a5/max_fs = 127
p7a5/max_ss = 3967
p7a5/res = 5000.0000000000
p7a5/coffset = 0.0000386666
p7a5/fs = -0.000100x -1.000000y
p7a5/ss = +1.000000x -0.000100y
p7a5/corner_x = -217.267
p7a5/corner_y = -464.904

p7a6/min_fs = 0
p7a6/min_ss = 3968
p7a6/max_fs = 127
p7a6/max_ss = 4031
p7a6/res = 5000.0000000000
p7a6/coffset = 0.0000353873
p7a6/fs = -0.000100x -1.000000y
p7a6/ss = +1.000000x -0.000100y
p7a6/corner_x = -151.732
p7a6/corner_y = -464.911

p7a7/min_fs = 0
p7a7/min_ss = 4032
p7a7/max_fs = 127
p7a7/max_ss = 4095
p7a7/res = 5000.0000000000
p7a7/coffset = 0.0000321079
p7a7/fs = -0.000100x -1.000000y
p7a7/ss = +1.000000x -0.000100y
p7a7/corner_x = -86.1973
p7a7/corner_y = -464.918

p8a0/min_fs = 0
p8a0/min_ss = 4096
p8a0/max_fs = 127
p8a0/max_ss = 4159
p8a0/res = 5000.0000000000
p8a0/coffset = 0.0000056537
p8a0/fs = -0.001629x +0.999995y
p8a0/ss = -0.999998x -0.001626y
p8a0/corner_x = 528.878
p8a0/corner_y = -162.63

p8a1/min_fs = 0
p8a1/min_ss = 4160
p8a1/max_fs = 127
p8a1/max_ss = 4223
p8a1/res = 5000.0000000000
p8a1/coffset = 0.0000178665
p8a1/fs = -0.001629x +0.999995y
p8a1/ss = -0.999998x -0.001626y
p8a1/corner_x = 463.343
p8a1/corner_y = -162.737

p8a2/min_fs = 0
p8a2/min_ss = 4224
p8a2/max_fs = 127
p8a2/max_ss = 4287
p8a2/res = 5000.0000000000
p8a2/coffset = 0.0000300793
p8a2/fs = -0.001629x +0.999995y
p8a2/ss = -0.999998x -0.001626y
p8a2/corner_x = 397.808
p8a2/corner_y = -162.843

p8a3/min_fs = 0
p8a3/min_ss = 4288
p8a3/max_fs = 127
p8a3/max_ss = 4351
p8a3/res = 5000.0000000000
p8a3/coffset = 0.0000422922
p8a3/fs = -0.001629x +0.999995y
p8a3/ss = -0.999998x -0.001626y
p8a3/corner_x = 332.273
p8a3/corner_y = -162.95

p8a4/min_fs = 0
p8a4/min_ss = 4352
p8a4/max_fs = 127
p8a4/max_ss = 4415
p8a4/res = 5000.0000000000
p8a4/coffset = 0.0000545048
p8a4/fs = -0.001629x +0.999995y
p8a4/ss = -0.999998x -0.001626y
p8a4/corner_x = 266.739
p8a4/corner_y = -163.056

p8a5/min_fs = 0
p8a5/min_ss = 4416
p8a5/max_fs = 127
p8a5/max_ss = 4479
p8a5/res = 5000.0000000000
p8a5/coffset = 0.0000667177
p8a5/fs = -0.001629x +0.999995y
p8a5/ss = -0.999998x -0.001626y
p8a5/corner_x = 201.204
p8a5/corner_y = -163.163

p8a6/min_fs = 0
p8a6/min_ss = 4480
p8a6/max_fs = 127
p8a6/max_ss = 4543
p8a6/res = 5000.0000000000
p8a6/coffset = 0.0000789305
p8a6/fs = -0.001629x +0.999995y
p8a6/ss = -0.999998x -0.001626y
p8a6/corner_x = 135.67
p8a6/corner_y = -163.269

p8a7/min_fs = 0
p8a7/min_ss = 4544
p8a7/max_fs = 127
p8a7/max_ss = 4607
p8a7/res = 5000.0000000000
p8a7/coffset = 0.0000911432
p8a7/fs = -0.001629x +0.999995y
p8a7/ss = -0.999998x -0.001626y
p8a7/corner_x = 70.1354
p8a7/corner_y = -163.376

p9a0/min_fs = 0
p9a0/min_ss = 4608
p9a0/max_fs = 127
p9a0/max_ss = 4671
p9a0/res = 5000.0000000000
p9a0/coffset = -0.0009574177
p9a0/fs = -0.000697x +1.000000y
p9a0/ss = -0.999999x -0.000697y
p9a0/corner_x = 527.343
p9a0/corner_y = -317.936

p9a1/min_fs = 0
p9a1/min_ss = 4672
p9a1/max_fs = 127
p9a1/max_ss = 4735
p9a1/res = 5000.0000000000
p9a1/coffset = -0.0009478795
p9a1/fs = -0.000697x +1.000000y
p9a1/ss = -0.999999x -0.000697y
p9a1/corner_x = 461.809
p9a1/corner_y = -317.981

p9a2/min_fs = 0
p9a2/min_ss = 4736
p9a2/max_fs = 127
p9a2/max_ss = 4799
p9a2/res = 5000.0000000000
p9a2/coffset = -0.0009383412
p9a2/fs = -0.000697x +1.000000y
p9a2/ss = -0.999999x -0.000697y
p9a2/corner_x = 396.274
p9a2/corner_y = -318.027

p9a3/min_fs = 0
p9a3/min_ss = 4800
p9a3/max_fs = 127
p9a3/max_ss = 4863
p9a3/res = 5000.0000000000
p9a3/coffset = -0.0009288029
p9a3/fs = -0.000697x +1.000000y
p9a3/ss = -0.999999x -0.000697y
p9a3/corner_x = 330.739
p9a3/corner_y = -318.073

p9a4/min_fs = 0
p9a4/min_ss = 4864
p9a4/max_fs = 127
p9a4/max_ss = 4927
p9a4/res = 5000.0000000000
p9a4/coffset = -0.0009192646
p9a4/fs = -0.000697x +1.000000y
p9a4/ss = -0.999999x -0.000697y
p9a4/corner_x = 265.204
p9a4/corner_y = -318.118

p9a5/min_fs = 0
p9a5/min_ss = 4928
p9a5/max_fs = 127
p9a5/max_ss = 4991
p9a5/res = 5000.0000000000
p9a5/coffset = -0.0009097264
p9a5/fs = -0.000697x +1.000000y
p9a5/ss = -0.999999x -0.000697y
p9a5/corner_x = 199.67
p9a5/corner_y = -318.164

p9a6/min_fs = 0
p9a6/min_ss = 4992
p9a6/max_fs = 127
p9a6/max_ss = 5055
p9a6/res = 5000.0000000000
p9a6/coffset = -0.0009001881
p9a6/fs = -0.000697x +1.000000y
p9a6/ss = -0.999999x -0.000697y
p9a6/corner_x = 134.135
p9a6/corner_y = -318.21

p9a7/min_fs = 0
p9a7/min_ss = 5056
p9a7/max_fs = 127
p9a7/max_ss = 5119
p9a7/res = 5000.0000000000
p9a7/coffset = -0.0008906498
p9a7/fs = -0.000697x +1.000000y
p9a7/ss = -0.999999x -0.000697y
p9a7/corner_x = 68.6001
p9a7/corner_y = -318.255

p10a0/min_fs = 0
p10a0/min_ss = 5120
p10a0/max_fs = 127
p10a0/max_ss = 5183
p10a0/res = 5000.0000000000
p10a0/coffset = -0.0004407863
p10a0/fs = -0.000016x +1.000000y
p10a0/ss = -1.000000x -0.000016y
p10a0/corner_x = 527.269
p10a0/corner_y = -475.248

p10a1/min_fs = 0
p10a1/min_ss = 5184
p10a1/max_fs = 127
p10a1/max_ss = 5247
p10a1/res = 5000.0000000000
p10a1/coffset = -0.0004321845
p10a1/fs = -0.000016x +1.000000y
p10a1/ss = -1.000000x -0.000016y
p10a1/corner_x = 461.734
p10a1/corner_y = -475.249

p10a2/min_fs = 0
p10a2/min_ss = 5248
p10a2/max_fs = 127
p10a2/max_ss = 5311
p10a2/res = 5000.0000000000
p10a2/coffset = -0.0004235828
p10a2/fs = -0.000016x +1.000000y
p10a2/ss = -1.000000x -0.000016y
p10a2/corner_x = 396.199
p10a2/corner_y = -475.25

p10a3/min_fs = 0
p10a3/min_ss = 5312
p10a3/max_fs = 127
p10a3/max_ss = 5375
p10a3/res = 5000.0000000000
p10a3/coffset = -0.0004149812
p10a3/fs = -0.000016x +1.000000y
p10a3/ss = -1.000000x -0.000016y
p10a3/corner_x = 330.665
p10a3/corner_y = -475.251

p10a4/min_fs = 0
p10a4/min_ss = 5376
p10a4/max_fs = 127
p10a4/max_ss = 5439
p10a4/res = 5000.0000000000
p10a4/coffset = -0.0004063794
p10a4/fs = -0.000016x +1.000000y
p10a4/ss = -1.000000x -0.000016y
p10a4/corner_x = 265.13
p10a4/corner_y = -475.252

p10a5/min_fs = 0
p10a5/min_ss = 5440
p10a5/max_fs = 127
p10a5/max_ss = 5503
p10a5/res = 5000.0000000000
p10a5/coffset = -0.0003977777
p10a5/fs = -0.000016x +1.000000y
p10a5/ss = -1.000000x -0.000016y
p10a5/corner_x = 199.595
p10a5/corner_y = -475.253

p10a6/min_fs = 0
p10a6/min_ss = 5504
p10a6/max_fs = 127
p10a6/max_ss = 5567
p10a6/res = 5000.0000000000
p10a6/coffset = -0.0003891759
p10a6/fs = -0.000016x +1.000000y
p10a6/ss = -1.000000x -0.000016y
p10a6/corner_x = 134.06
p10a6/corner_y = -475.254

p10a7/min_fs = 0
p10a7/min_ss = 5568
p10a7/max_fs = 127
p10a7/max_ss = 5631
p10a7/res = 5000.0000000000
p10a7/coffset = -0.0003805743
p10a7/fs = -0.000016x +1.000000y
p10a7/ss = -1.000000x -0.000016y
p10a7/corner_x = 68.5262
p10a7/corner_y = -475.255

p11a0/min_fs = 0
p11a0/min_ss = 5632
p11a0/max_fs = 127
p11a0/max_ss = 5695
p11a0/res = 5000.0000000000
p11a0/coffset = 0.0011897539
p11a0/fs = -0.001178x +0.999998y
p11a0/ss = -0.999999x -0.001179y
p11a0/corner_x = 529.837
p11a0/corner_y = -635.65

p11a1/min_fs = 0
p11a1/min_ss = 5696
p11a1/max_fs = 127
p11a1/max_ss = 5759
p11a1/res = 5000.0000000000
p11a1/coffset = 0.0011852551
p11a1/fs = -0.001178x +0.999998y
p11a1/ss = -0.999999x -0.001179y
p11a1/corner_x = 464.303
p11a1/corner_y = -635.728

p11a2/min_fs = 0
p11a2/min_ss = 5760
p11a2/max_fs = 127
p11a2/max_ss = 5823
p11a2/res = 5000.0000000000
p11a2/coffset = 0.0011807562
p11a2/fs = -0.001178x +0.999998y
p11a2/ss = -0.999999x -0.001179y
p11a2/corner_x = 398.768
p11a2/corner_y = -635.805

p11a3/min_fs = 0
p11a3/min_ss = 5824
p11a3/max_fs = 127
p11a3/max_ss = 5887
p11a3/res = 5000.0000000000
p11a3/coffset = 0.0011762574
p11a3/fs = -0.001178x +0.999998y
p11a3/ss = -0.999999x -0.001179y
p11a3/corner_x = 333.233
p11a3/corner_y = -635.882

p11a4/min_fs = 0
p11a4/min_ss = 5888
p11a4/max_fs = 127
p11a4/max_ss = 5951
p11a4/res = 5000.0000000000
p11a4/coffset = 0.0011717585
p11a4/fs = -0.001178x +0.999998y
p11a4/ss = -0.999999x -0.001179y
p11a4/corner_x = 267.698
p11a4/corner_y = -635.959

p11a5/min_fs = 0
p11a5/min_ss = 5952
p11a5/max_fs = 127
p11a5/max_ss = 6015
p11a5/res = 5000.0000000000
p11a5/coffset = 0.0011672597
p11a5/fs = -0.001178x +0.999998y
p11a5/ss = -0.999999x -0.001179y
p11a5/corner_x = 202.163
p11a5/corner_y = -636.037

p11a6/min_fs = 0
p11a6/min_ss = 6016
p11a6/max_fs = 127
p11a6/max_ss = 6079
p11a6/res = 5000.0000000000
p11a6/coffset = 0.0011627609
p11a6/fs = -0.001178x +0.999998y
p11a6/ss = -0.999999x -0.001179y
p11a6/corner_x = 136.629
p11a6/corner_y = -636.114

p11a7/min_fs = 0
p11a7/min_ss = 6080
p11a7/max_fs = 127
p11a7/max_ss = 6143
p11a7/res = 5000.0000000000
p11a7/coffset = 0.0011582621
p11a7/fs = -0.001178x +0.999998y
p11a7/ss = -0.999999x -0.001179y
p11a7/corner_x = 71.0944
p11a7/corner_y = -636.191

p12a0/min_fs = 0
p12a0/min_ss = 6144
p12a0/max_fs = 127
p12a0/max_ss = 6207
p12a0/res = 5000.0000000000
p12a0/coffset = -0.0000008928
p12a0/fs = -0.000343x +1.000000y
p12a0/ss = -1.000000x -0.000343y
p12a0/corner_x = 538.944
p12a0/corner_y = 485.939

p12a1/min_fs = 0
p12a1/min_ss = 6208
p12a1/max_fs = 127
p12a1/max_ss = 6271
p12a1/res = 5000.0000000000
p12a1/coffset = -0.0000054641
p12a1/fs = -0.000343x +1.000000y
p12a1/ss = -1.000000x -0.000343y
p12a1/corner_x = 473.409
p12a1/corner_y = 485.917

p12a2/min_fs = 0
p12a2/min_ss = 6272
p12a2/max_fs = 127
p12a2/max_ss = 6335
p12a2/res = 5000.0000000000
p12a2/coffset = -0.0000100354
p12a2/fs = -0.000343x +1.000000y
p12a2/ss = -1.000000x -0.000343y
p12a2/corner_x = 407.874
p12a2/corner_y = 485.894

p12a3/min_fs = 0
p12a3/min_ss = 6336
p12a3/max_fs = 127
p12a3/max_ss = 6399
p12a3/res = 5000.0000000000
p12a3/coffset = -0.0000146066
p12a3/fs = -0.000343x +1.000000y
p12a3/ss = -1.000000x -0.000343y
p12a3/corner_x = 342.339
p12a3/corner_y = 485.872

p12a4/min_fs = 0
p12a4/min_ss = 6400
p12a4/max_fs = 127
p12a4/max_ss = 6463
p12a4/res = 5000.0000000000
p12a4/coffset = -0.0000191779
p12a4/fs = -0.000343x +1.000000y
p12a4/ss = -1.000000x -0.000343y
p12a4/corner_x = 276.805
p12a4/corner_y = 485.849

p12a5/min_fs = 0
p12a5/min_ss = 6464
p12a5/max_fs = 127
p12a5/max_ss = 6527
p12a5/res = 5000.0000000000
p12a5/coffset = -0.0000237492
p12a5/fs = -0.000343x +1.000000y
p12a5/ss = -1.000000x -0.000343y
p12a5/corner_x = 211.27
p12a5/corner_y = 485.827

p12a6/min_fs = 0
p12a6/min_ss = 6528
p12a6/max_fs = 127
p12a6/max_ss = 6591
p12a6/res = 5000.0000000000
p12a6/coffset = -0.0000283205
p12a6/fs = -0.000343x +1.000000y
p12a6/ss = -1.000000x -0.000343y
p12a6/corner_x = 145.735
p12a6/corner_y = 485.804

p12a7/min_fs = 0
p12a7/min_ss = 6592
p12a7/max_fs = 127
p12a7/max_ss = 6655
p12a7/res = 5000.0000000000
p12a7/coffset = -0.0000328917
p12a7/fs = -0.000343x +1.000000y
p12a7/ss = -1.000000x -0.000343y
p12a7/corner_x = 80.2002
p12a7/corner_y = 485.782

p13a0/min_fs = 0
p13a0/min_ss = 6656
p13a0/max_fs = 127
p13a0/max_ss = 6719
p13a0/res = 5000.0000000000
p13a0/coffset = -0.0010368718
p13a0/fs = +0.000326x +1.000000y
p13a0/ss = -0.999999x +0.000326y
p13a0/corner_x = 538.692
p13a0/corner_y = 326.008

p13a1/min_fs = 0
p13a1/min_ss = 6720
p13a1/max_fs = 127
p13a1/max_ss = 6783
p13a1/res = 5000.0000000000
p13a1/coffset = -0.0010192014
p13a1/fs = +0.000326x +1.000000y
p13a1/ss = -0.999999x +0.000326y
p13a1/corner_x = 473.157
p13a1/corner_y = 326.03

p13a2/min_fs = 0
p13a2/min_ss = 6784
p13a2/max_fs = 127
p13a2/max_ss = 6847
p13a2/res = 5000.0000000000
p13a2/coffset = -0.0010015312
p13a2/fs = +0.000326x +1.000000y
p13a2/ss = -0.999999x +0.000326y
p13a2/corner_x = 407.623
p13a2/corner_y = 326.051

p13a3/min_fs = 0
p13a3/min_ss = 6848
p13a3/max_fs = 127
p13a3/max_ss = 6911
p13a3/res = 5000.0000000000
p13a3/coffset = -0.0009838608
p13a3/fs = +0.000326x +1.000000y
p13a3/ss = -0.999999x +0.000326y
p13a3/corner_x = 342.088
p13a3/corner_y = 326.072

p13a4/min_fs = 0
p13a4/min_ss = 6912
p13a4/max_fs = 127
p13a4/max_ss = 6975
p13a4/res = 5000.0000000000
p13a4/coffset = -0.0009661903
p13a4/fs = +0.000326x +1.000000y
p13a4/ss = -0.999999x +0.000326y
p13a4/corner_x = 276.553
p13a4/corner_y = 326.094

p13a5/min_fs = 0
p13a5/min_ss = 6976
p13a5/max_fs = 127
p13a5/max_ss = 7039
p13a5/res = 5000.0000000000
p13a5/coffset = -0.0009485198
p13a5/fs = +0.000326x +1.000000y
p13a5/ss = -0.999999x +0.000326y
p13a5/corner_x = 211.018
p13a5/corner_y = 326.115

p13a6/min_fs = 0
p13a6/min_ss = 7040
p13a6/max_fs = 127
p13a6/max_ss = 7103
p13a6/res = 5000.0000000000
p13a6/coffset = -0.0009308497
p13a6/fs = +0.000326x +1.000000y
p13a6/ss = -0.999999x +0.000326y
p13a6/corner_x = 145.484
p13a6/corner_y = 326.137

p13a7/min_fs = 0
p13a7/min_ss = 7104
p13a7/max_fs = 127
p13a7/max_ss = 7167
p13a7/res = 5000.0000000000
p13a7/coffset = -0.0009131792
p13a7/fs = +0.000326x +1.000000y
p13a7/ss = -0.999999x +0.000326y
p13a7/corner_x = 79.9494
p13a7/corner_y = 326.158

p14a0/min_fs = 0
p14a0/min_ss = 7168
p14a0/max_fs = 127
p14a0/max_ss = 7231
p14a0/res = 5000.0000000000
p14a0/coffset = -0.0008535362
p14a0/fs = +0.000486x +1.000000y
p14a0/ss = -1.000000x +0.000486y
p14a0/corner_x = 538.846
p14a0/corner_y = 169.369

p14a1/min_fs = 0
p14a1/min_ss = 7232
p14a1/max_fs = 127
p14a1/max_ss = 7295
p14a1/res = 5000.0000000000
p14a1/coffset = -0.0008456216
p14a1/fs = +0.000486x +1.000000y
p14a1/ss = -1.000000x +0.000486y
p14a1/corner_x = 473.311
p14a1/corner_y = 169.401

p14a2/min_fs = 0
p14a2/min_ss = 7296
p14a2/max_fs = 127
p14a2/max_ss = 7359
p14a2/res = 5000.0000000000
p14a2/coffset = -0.0008377071
p14a2/fs = +0.000486x +1.000000y
p14a2/ss = -1.000000x +0.000486y
p14a2/corner_x = 407.776
p14a2/corner_y = 169.433

p14a3/min_fs = 0
p14a3/min_ss = 7360
p14a3/max_fs = 127
p14a3/max_ss = 7423
p14a3/res = 5000.0000000000
p14a3/coffset = -0.0008297925
p14a3/fs = +0.000486x +1.000000y
p14a3/ss = -1.000000x +0.000486y
p14a3/corner_x = 342.241
p14a3/corner_y = 169.465

p14a4/min_fs = 0
p14a4/min_ss = 7424
p14a4/max_fs = 127
p14a4/max_ss = 7487
p14a4/res = 5000.0000000000
p14a4/coffset = -0.0008218781
p14a4/fs = +0.000486x +1.000000y
p14a4/ss = -1.000000x +0.000486y
p14a4/corner_x = 276.707
p14a4/corner_y = 169.497

p14a5/min_fs = 0
p14a5/min_ss = 7488
p14a5/max_fs = 127
p14a5/max_ss = 7551
p14a5/res = 5000.0000000000
p14a5/coffset = -0.0008139635
p14a5/fs = +0.000486x +1.000000y
p14a5/ss = -1.000000x +0.000486y
p14a5/corner_x = 211.172
p14a5/corner_y = 169.529

p14a6/min_fs = 0
p14a6/min_ss = 7552
p14a6/max_fs = 127
p14a6/max_ss = 7615
p14a6/res = 5000.0000000000
p14a6/coffset = -0.0008060489
p14a6/fs = +0.000486x +1.000000y
p14a6/ss = -1.000000x +0.000486y
p14a6/corner_x = 145.637
p14a6/corner_y = 169.561

p14a7/min_fs = 0
p14a7/min_ss = 7616
p14a7/max_fs = 127
p14a7/max_ss = 7679
p14a7/res = 5000.0000000000
p14a7/coffset = -0.0007981344
p14a7/fs = +0.000486x +1.000000y
p14a7/ss = -1.000000x +0.000486y
p14a7/corner_x = 80.1023
p14a7/corner_y = 169.592

p15a0/min_fs = 0
p15a0/min_ss = 7680
p15a0/max_fs = 127
p15a0/max_ss = 7743
p15a0/res = 5000.0000000000
p15a0/coffset = -0.0013929242
p15a0/fs = -0.000121x +0.999999y
p15a0/ss = -1.000000x -0.000121y
p15a0/corner_x = 537.843
p15a0/corner_y = 13.8552

p15a1/min_fs = 0
p15a1/min_ss = 7744
p15a1/max_fs = 127
p15a1/max_ss = 7807
p15a1/res = 5000.0000000000
p15a1/coffset = -0.0013897634
p15a1/fs = -0.000121x +0.999999y
p15a1/ss = -1.000000x -0.000121y
p15a1/corner_x = 472.308
p15a1/corner_y = 13.8473

p15a2/min_fs = 0
p15a2/min_ss = 7808
p15a2/max_fs = 127
p15a2/max_ss = 7871
p15a2/res = 5000.0000000000
p15a2/coffset = -0.0013866027
p15a2/fs = -0.000121x +0.999999y
p15a2/ss = -1.000000x -0.000121y
p15a2/corner_x = 406.774
p15a2/corner_y = 13.8394

p15a3/min_fs = 0
p15a3/min_ss = 7872
p15a3/max_fs = 127
p15a3/max_ss = 7935
p15a3/res = 5000.0000000000
p15a3/coffset = -0.0013834419
p15a3/fs = -0.000121x +0.999999y
p15a3/ss = -1.000000x -0.000121y
p15a3/corner_x = 341.239
p15a3/corner_y = 13.8314

p15a4/min_fs = 0
p15a4/min_ss = 7936
p15a4/max_fs = 127
p15a4/max_ss = 7999
p15a4/res = 5000.0000000000
p15a4/coffset = -0.0013802811
p15a4/fs = -0.000121x +0.999999y
p15a4/ss = -1.000000x -0.000121y
p15a4/corner_x = 275.704
p15a4/corner_y = 13.8235

p15a5/min_fs = 0
p15a5/min_ss = 8000
p15a5/max_fs = 127
p15a5/max_ss = 8063
p15a5/res = 5000.0000000000
p15a5/coffset = -0.0013771204
p15a5/fs = -0.000121x +0.999999y
p15a5/ss = -1.000000x -0.000121y
p15a5/corner_x = 210.169
p15a5/corner_y = 13.8155

p15a6/min_fs = 0
p15a6/min_ss = 8064
p15a6/max_fs = 127
p15a6/max_ss = 8127
p15a6/res = 5000.0000000000
p15a6/coffset = -0.0013739596
p15a6/fs = -0.000121x +0.999999y
p15a6/ss = -1.000000x -0.000121y
p15a6/corner_x = 144.635
p15a6/corner_y = 13.8076

p15a7/min_fs = 0
p15a7/min_ss = 8128
p15a7/max_fs = 127
p15a7/max_ss = 8191
p15a7/res = 5000.0000000000
p15a7/coffset = -0.0013707988
p15a7/fs = -0.000121x +0.999999y
p15a7/ss = -1.000000x -0.000121y
p15a7/corner_x = 79.0998
p15a7/corner_y = 13.7996

