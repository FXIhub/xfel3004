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
p0a0/max_fs = 127
p0a0/min_ss = 0
p0a0/max_ss = 63
p0a0/fs = +0.001918x -0.999997y
p0a0/ss = +0.999996x +0.001915y
p0a0/corner_x = -533.5584852095981
p0a0/corner_y = 626.6121010394062

p0a1/min_fs = 0
p0a1/max_fs = 127
p0a1/min_ss = 64
p0a1/max_ss = 127
p0a1/fs = +0.001918x -0.999997y
p0a1/ss = +0.999996x +0.001915y
p0a1/corner_x = -468.0234852095981
p0a1/corner_y = 626.7371010394062

p0a2/min_fs = 0
p0a2/max_fs = 127
p0a2/min_ss = 128
p0a2/max_ss = 191
p0a2/fs = +0.001918x -0.999997y
p0a2/ss = +0.999996x +0.001915y
p0a2/corner_x = -402.49048520959815
p0a2/corner_y = 626.8631010394063

p0a3/min_fs = 0
p0a3/max_fs = 127
p0a3/min_ss = 192
p0a3/max_ss = 255
p0a3/fs = +0.001918x -0.999997y
p0a3/ss = +0.999996x +0.001915y
p0a3/corner_x = -336.9554852095981
p0a3/corner_y = 626.9881010394063

p0a4/min_fs = 0
p0a4/max_fs = 127
p0a4/min_ss = 256
p0a4/max_ss = 319
p0a4/fs = +0.001918x -0.999997y
p0a4/ss = +0.999996x +0.001915y
p0a4/corner_x = -271.42048520959816
p0a4/corner_y = 627.1141010394063

p0a5/min_fs = 0
p0a5/max_fs = 127
p0a5/min_ss = 320
p0a5/max_ss = 383
p0a5/fs = +0.001918x -0.999997y
p0a5/ss = +0.999996x +0.001915y
p0a5/corner_x = -205.88548520959813
p0a5/corner_y = 627.2391010394063

p0a6/min_fs = 0
p0a6/max_fs = 127
p0a6/min_ss = 384
p0a6/max_ss = 447
p0a6/fs = +0.001918x -0.999997y
p0a6/ss = +0.999996x +0.001915y
p0a6/corner_x = -140.35248520959814
p0a6/corner_y = 627.3651010394062

p0a7/min_fs = 0
p0a7/max_fs = 127
p0a7/min_ss = 448
p0a7/max_ss = 511
p0a7/fs = +0.001918x -0.999997y
p0a7/ss = +0.999996x +0.001915y
p0a7/corner_x = -74.81668520959815
p0a7/corner_y = 627.4901010394062

p1a0/min_fs = 0
p1a0/max_fs = 127
p1a0/min_ss = 512
p1a0/max_ss = 575
p1a0/fs = +0.000841x -0.999997y
p1a0/ss = +0.999999x +0.000838y
p1a0/corner_x = -532.6894852095982
p1a0/corner_y = 470.8941010394062

p1a1/min_fs = 0
p1a1/max_fs = 127
p1a1/min_ss = 576
p1a1/max_ss = 639
p1a1/fs = +0.000841x -0.999997y
p1a1/ss = +0.999999x +0.000838y
p1a1/corner_x = -467.1554852095981
p1a1/corner_y = 470.94910103940623

p1a2/min_fs = 0
p1a2/max_fs = 127
p1a2/min_ss = 640
p1a2/max_ss = 703
p1a2/fs = +0.000841x -0.999997y
p1a2/ss = +0.999999x +0.000838y
p1a2/corner_x = -401.62048520959814
p1a2/corner_y = 471.00310103940626

p1a3/min_fs = 0
p1a3/max_fs = 127
p1a3/min_ss = 704
p1a3/max_ss = 767
p1a3/fs = +0.000841x -0.999997y
p1a3/ss = +0.999999x +0.000838y
p1a3/corner_x = -336.0854852095981
p1a3/corner_y = 471.0581010394062

p1a4/min_fs = 0
p1a4/max_fs = 127
p1a4/min_ss = 768
p1a4/max_ss = 831
p1a4/fs = +0.000841x -0.999997y
p1a4/ss = +0.999999x +0.000838y
p1a4/corner_x = -270.55048520959815
p1a4/corner_y = 471.1131010394062

p1a5/min_fs = 0
p1a5/max_fs = 127
p1a5/min_ss = 832
p1a5/max_ss = 895
p1a5/fs = +0.000841x -0.999997y
p1a5/ss = +0.999999x +0.000838y
p1a5/corner_x = -205.01648520959813
p1a5/corner_y = 471.1681010394062

p1a6/min_fs = 0
p1a6/max_fs = 127
p1a6/min_ss = 896
p1a6/max_ss = 959
p1a6/fs = +0.000841x -0.999997y
p1a6/ss = +0.999999x +0.000838y
p1a6/corner_x = -139.48148520959813
p1a6/corner_y = 471.22310103940623

p1a7/min_fs = 0
p1a7/max_fs = 127
p1a7/min_ss = 960
p1a7/max_ss = 1023
p1a7/fs = +0.000841x -0.999997y
p1a7/ss = +0.999999x +0.000838y
p1a7/corner_x = -73.94668520959814
p1a7/corner_y = 471.27810103940624

p2a0/min_fs = 0
p2a0/max_fs = 127
p2a0/min_ss = 1024
p2a0/max_ss = 1087
p2a0/fs = +0.000489x -0.999998y
p2a0/ss = +1.000000x +0.000488y
p2a0/corner_x = -531.5334852095981
p2a0/corner_y = 314.7551010394062

p2a1/min_fs = 0
p2a1/max_fs = 127
p2a1/min_ss = 1088
p2a1/max_ss = 1151
p2a1/fs = +0.000489x -0.999998y
p2a1/ss = +1.000000x +0.000488y
p2a1/corner_x = -465.9994852095981
p2a1/corner_y = 314.78710103940625

p2a2/min_fs = 0
p2a2/max_fs = 127
p2a2/min_ss = 1152
p2a2/max_ss = 1215
p2a2/fs = +0.000489x -0.999998y
p2a2/ss = +1.000000x +0.000488y
p2a2/corner_x = -400.46448520959814
p2a2/corner_y = 314.81910103940623

p2a3/min_fs = 0
p2a3/max_fs = 127
p2a3/min_ss = 1216
p2a3/max_ss = 1279
p2a3/fs = +0.000489x -0.999998y
p2a3/ss = +1.000000x +0.000488y
p2a3/corner_x = -334.93048520959815
p2a3/corner_y = 314.8511010394062

p2a4/min_fs = 0
p2a4/max_fs = 127
p2a4/min_ss = 1280
p2a4/max_ss = 1343
p2a4/fs = +0.000489x -0.999998y
p2a4/ss = +1.000000x +0.000488y
p2a4/corner_x = -269.3954852095981
p2a4/corner_y = 314.88310103940626

p2a5/min_fs = 0
p2a5/max_fs = 127
p2a5/min_ss = 1344
p2a5/max_ss = 1407
p2a5/fs = +0.000489x -0.999998y
p2a5/ss = +1.000000x +0.000488y
p2a5/corner_x = -203.86148520959813
p2a5/corner_y = 314.91510103940624

p2a6/min_fs = 0
p2a6/max_fs = 127
p2a6/min_ss = 1408
p2a6/max_ss = 1471
p2a6/fs = +0.000489x -0.999998y
p2a6/ss = +1.000000x +0.000488y
p2a6/corner_x = -138.32648520959813
p2a6/corner_y = 314.9471010394062

p2a7/min_fs = 0
p2a7/max_fs = 127
p2a7/min_ss = 1472
p2a7/max_ss = 1535
p2a7/fs = +0.000489x -0.999998y
p2a7/ss = +1.000000x +0.000488y
p2a7/corner_x = -72.79088520959814
p2a7/corner_y = 314.97910103940626

p3a0/min_fs = 0
p3a0/max_fs = 127
p3a0/min_ss = 1536
p3a0/max_ss = 1599
p3a0/fs = +0.001241x -0.999999y
p3a0/ss = +0.999999x +0.001241y
p3a0/corner_x = -531.0814852095981
p3a0/corner_y = 158.08810103940624

p3a1/min_fs = 0
p3a1/max_fs = 127
p3a1/min_ss = 1600
p3a1/max_ss = 1663
p3a1/fs = +0.001241x -0.999999y
p3a1/ss = +0.999999x +0.001241y
p3a1/corner_x = -465.5474852095981
p3a1/corner_y = 158.16910103940626

p3a2/min_fs = 0
p3a2/max_fs = 127
p3a2/min_ss = 1664
p3a2/max_ss = 1727
p3a2/fs = +0.001241x -0.999999y
p3a2/ss = +0.999999x +0.001241y
p3a2/corner_x = -400.01248520959814
p3a2/corner_y = 158.25010103940627

p3a3/min_fs = 0
p3a3/max_fs = 127
p3a3/min_ss = 1728
p3a3/max_ss = 1791
p3a3/fs = +0.001241x -0.999999y
p3a3/ss = +0.999999x +0.001241y
p3a3/corner_x = -334.4774852095981
p3a3/corner_y = 158.33210103940627

p3a4/min_fs = 0
p3a4/max_fs = 127
p3a4/min_ss = 1792
p3a4/max_ss = 1855
p3a4/fs = +0.001241x -0.999999y
p3a4/ss = +0.999999x +0.001241y
p3a4/corner_x = -268.94248520959815
p3a4/corner_y = 158.41310103940623

p3a5/min_fs = 0
p3a5/max_fs = 127
p3a5/min_ss = 1856
p3a5/max_ss = 1919
p3a5/fs = +0.001241x -0.999999y
p3a5/ss = +0.999999x +0.001241y
p3a5/corner_x = -203.40848520959813
p3a5/corner_y = 158.49410103940625

p3a6/min_fs = 0
p3a6/max_fs = 127
p3a6/min_ss = 1920
p3a6/max_ss = 1983
p3a6/fs = +0.001241x -0.999999y
p3a6/ss = +0.999999x +0.001241y
p3a6/corner_x = -137.87348520959813
p3a6/corner_y = 158.57610103940624

p3a7/min_fs = 0
p3a7/max_fs = 127
p3a7/min_ss = 1984
p3a7/max_ss = 2047
p3a7/fs = +0.001241x -0.999999y
p3a7/ss = +0.999999x +0.001241y
p3a7/corner_x = -72.33938520959815
p3a7/corner_y = 158.65710103940626

p4a0/min_fs = 0
p4a0/max_fs = 127
p4a0/min_ss = 2048
p4a0/max_ss = 2111
p4a0/fs = -0.000185x -1.000000y
p4a0/ss = +1.000000x -0.000185y
p4a0/corner_x = -543.7305299993164
p4a0/corner_y = -2.865375097609863

p4a1/min_fs = 0
p4a1/max_fs = 127
p4a1/min_ss = 2112
p4a1/max_ss = 2175
p4a1/fs = -0.000185x -1.000000y
p4a1/ss = +1.000000x -0.000185y
p4a1/corner_x = -478.1955299993164
p4a1/corner_y = -2.877515097609863

p4a2/min_fs = 0
p4a2/max_fs = 127
p4a2/min_ss = 2176
p4a2/max_ss = 2239
p4a2/fs = -0.000185x -1.000000y
p4a2/ss = +1.000000x -0.000185y
p4a2/corner_x = -412.66052999931645
p4a2/corner_y = -2.889655097609863

p4a3/min_fs = 0
p4a3/max_fs = 127
p4a3/min_ss = 2240
p4a3/max_ss = 2303
p4a3/fs = -0.000185x -1.000000y
p4a3/ss = +1.000000x -0.000185y
p4a3/corner_x = -347.1255299993164
p4a3/corner_y = -2.901795097609863

p4a4/min_fs = 0
p4a4/max_fs = 127
p4a4/min_ss = 2304
p4a4/max_ss = 2367
p4a4/fs = -0.000185x -1.000000y
p4a4/ss = +1.000000x -0.000185y
p4a4/corner_x = -281.59152999931644
p4a4/corner_y = -2.9139350976098632

p4a5/min_fs = 0
p4a5/max_fs = 127
p4a5/min_ss = 2368
p4a5/max_ss = 2431
p4a5/fs = -0.000185x -1.000000y
p4a5/ss = +1.000000x -0.000185y
p4a5/corner_x = -216.0565299993164
p4a5/corner_y = -2.9260750976098633

p4a6/min_fs = 0
p4a6/max_fs = 127
p4a6/min_ss = 2432
p4a6/max_ss = 2495
p4a6/fs = -0.000185x -1.000000y
p4a6/ss = +1.000000x -0.000185y
p4a6/corner_x = -150.5215299993164
p4a6/corner_y = -2.9382150976098633

p4a7/min_fs = 0
p4a7/max_fs = 127
p4a7/min_ss = 2496
p4a7/max_ss = 2559
p4a7/fs = -0.000185x -1.000000y
p4a7/ss = +1.000000x -0.000185y
p4a7/corner_x = -84.9870299993164
p4a7/corner_y = -2.9503550976098634

p5a0/min_fs = 0
p5a0/max_fs = 127
p5a0/min_ss = 2560
p5a0/max_ss = 2623
p5a0/fs = +0.000048x -1.000000y
p5a0/ss = +1.000000x +0.000048y
p5a0/corner_x = -544.2205299993165
p5a0/corner_y = -160.47155509760987

p5a1/min_fs = 0
p5a1/max_fs = 127
p5a1/min_ss = 2624
p5a1/max_ss = 2687
p5a1/fs = +0.000048x -1.000000y
p5a1/ss = +1.000000x +0.000048y
p5a1/corner_x = -478.68552999931643
p5a1/corner_y = -160.46855509760988

p5a2/min_fs = 0
p5a2/max_fs = 127
p5a2/min_ss = 2688
p5a2/max_ss = 2751
p5a2/fs = +0.000048x -1.000000y
p5a2/ss = +1.000000x +0.000048y
p5a2/corner_x = -413.1505299993164
p5a2/corner_y = -160.46455509760986

p5a3/min_fs = 0
p5a3/max_fs = 127
p5a3/min_ss = 2752
p5a3/max_ss = 2815
p5a3/fs = +0.000048x -1.000000y
p5a3/ss = +1.000000x +0.000048y
p5a3/corner_x = -347.61552999931644
p5a3/corner_y = -160.46155509760987

p5a4/min_fs = 0
p5a4/max_fs = 127
p5a4/min_ss = 2816
p5a4/max_ss = 2879
p5a4/fs = +0.000048x -1.000000y
p5a4/ss = +1.000000x +0.000048y
p5a4/corner_x = -282.08152999931644
p5a4/corner_y = -160.45855509760986

p5a5/min_fs = 0
p5a5/max_fs = 127
p5a5/min_ss = 2880
p5a5/max_ss = 2943
p5a5/fs = +0.000048x -1.000000y
p5a5/ss = +1.000000x +0.000048y
p5a5/corner_x = -216.5465299993164
p5a5/corner_y = -160.45555509760987

p5a6/min_fs = 0
p5a6/max_fs = 127
p5a6/min_ss = 2944
p5a6/max_ss = 3007
p5a6/fs = +0.000048x -1.000000y
p5a6/ss = +1.000000x +0.000048y
p5a6/corner_x = -151.0115299993164
p5a6/corner_y = -160.45255509760986

p5a7/min_fs = 0
p5a7/max_fs = 127
p5a7/min_ss = 3008
p5a7/max_ss = 3071
p5a7/fs = +0.000048x -1.000000y
p5a7/ss = +1.000000x +0.000048y
p5a7/corner_x = -85.4770299993164
p5a7/corner_y = -160.44955509760987

p6a0/min_fs = 0
p6a0/max_fs = 127
p6a0/min_ss = 3072
p6a0/max_ss = 3135
p6a0/fs = +0.000039x -1.000000y
p6a0/ss = +0.999999x +0.000038y
p6a0/corner_x = -544.7195299993165
p6a0/corner_y = -317.99955509760986

p6a1/min_fs = 0
p6a1/max_fs = 127
p6a1/min_ss = 3136
p6a1/max_ss = 3199
p6a1/fs = +0.000039x -1.000000y
p6a1/ss = +0.999999x +0.000038y
p6a1/corner_x = -479.18452999931645
p6a1/corner_y = -317.99755509760985

p6a2/min_fs = 0
p6a2/max_fs = 127
p6a2/min_ss = 3200
p6a2/max_ss = 3263
p6a2/fs = +0.000039x -1.000000y
p6a2/ss = +0.999999x +0.000038y
p6a2/corner_x = -413.6505299993164
p6a2/corner_y = -317.99455509760986

p6a3/min_fs = 0
p6a3/max_fs = 127
p6a3/min_ss = 3264
p6a3/max_ss = 3327
p6a3/fs = +0.000039x -1.000000y
p6a3/ss = +0.999999x +0.000038y
p6a3/corner_x = -348.11552999931644
p6a3/corner_y = -317.99255509760985

p6a4/min_fs = 0
p6a4/max_fs = 127
p6a4/min_ss = 3328
p6a4/max_ss = 3391
p6a4/fs = +0.000039x -1.000000y
p6a4/ss = +0.999999x +0.000038y
p6a4/corner_x = -282.5805299993164
p6a4/corner_y = -317.98955509760987

p6a5/min_fs = 0
p6a5/max_fs = 127
p6a5/min_ss = 3392
p6a5/max_ss = 3455
p6a5/fs = +0.000039x -1.000000y
p6a5/ss = +0.999999x +0.000038y
p6a5/corner_x = -217.04552999931641
p6a5/corner_y = -317.98755509760986

p6a6/min_fs = 0
p6a6/max_fs = 127
p6a6/min_ss = 3456
p6a6/max_ss = 3519
p6a6/fs = +0.000039x -1.000000y
p6a6/ss = +0.999999x +0.000038y
p6a6/corner_x = -151.5115299993164
p6a6/corner_y = -317.98455509760987

p6a7/min_fs = 0
p6a7/max_fs = 127
p6a7/min_ss = 3520
p6a7/max_ss = 3583
p6a7/fs = +0.000039x -1.000000y
p6a7/ss = +0.999999x +0.000038y
p6a7/corner_x = -85.9769299993164
p6a7/corner_y = -317.98255509760986

p7a0/min_fs = 0
p7a0/max_fs = 127
p7a0/min_ss = 3584
p7a0/max_ss = 3647
p7a0/fs = -0.000100x -1.000000y
p7a0/ss = +1.000000x -0.000100y
p7a0/corner_x = -544.3635299993165
p7a0/corner_y = -471.3925550976099

p7a1/min_fs = 0
p7a1/max_fs = 127
p7a1/min_ss = 3648
p7a1/max_ss = 3711
p7a1/fs = -0.000100x -1.000000y
p7a1/ss = +1.000000x -0.000100y
p7a1/corner_x = -478.82952999931643
p7a1/corner_y = -471.39855509760986

p7a2/min_fs = 0
p7a2/max_fs = 127
p7a2/min_ss = 3712
p7a2/max_ss = 3775
p7a2/fs = -0.000100x -1.000000y
p7a2/ss = +1.000000x -0.000100y
p7a2/corner_x = -413.2945299993164
p7a2/corner_y = -471.40555509760986

p7a3/min_fs = 0
p7a3/max_fs = 127
p7a3/min_ss = 3776
p7a3/max_ss = 3839
p7a3/fs = -0.000100x -1.000000y
p7a3/ss = +1.000000x -0.000100y
p7a3/corner_x = -347.75952999931644
p7a3/corner_y = -471.4115550976099

p7a4/min_fs = 0
p7a4/max_fs = 127
p7a4/min_ss = 3840
p7a4/max_ss = 3903
p7a4/fs = -0.000100x -1.000000y
p7a4/ss = +1.000000x -0.000100y
p7a4/corner_x = -282.2245299993164
p7a4/corner_y = -471.4185550976099

p7a5/min_fs = 0
p7a5/max_fs = 127
p7a5/min_ss = 3904
p7a5/max_ss = 3967
p7a5/fs = -0.000100x -1.000000y
p7a5/ss = +1.000000x -0.000100y
p7a5/corner_x = -216.6905299993164
p7a5/corner_y = -471.42455509760987

p7a6/min_fs = 0
p7a6/max_fs = 127
p7a6/min_ss = 3968
p7a6/max_ss = 4031
p7a6/fs = -0.000100x -1.000000y
p7a6/ss = +1.000000x -0.000100y
p7a6/corner_x = -151.1555299993164
p7a6/corner_y = -471.4315550976099

p7a7/min_fs = 0
p7a7/max_fs = 127
p7a7/min_ss = 4032
p7a7/max_ss = 4095
p7a7/fs = -0.000100x -1.000000y
p7a7/ss = +1.000000x -0.000100y
p7a7/corner_x = -85.6208299993164
p7a7/corner_y = -471.4385550976099

p8a0/min_fs = 0
p8a0/max_fs = 127
p8a0/min_ss = 4096
p8a0/max_ss = 4159
p8a0/fs = -0.001629x +0.999995y
p8a0/ss = -0.999998x -0.001626y
p8a0/corner_x = 528.3513575643765
p8a0/corner_y = -164.0565206696992

p8a1/min_fs = 0
p8a1/max_fs = 127
p8a1/min_ss = 4160
p8a1/max_ss = 4223
p8a1/fs = -0.001629x +0.999995y
p8a1/ss = -0.999998x -0.001626y
p8a1/corner_x = 462.81635756437646
p8a1/corner_y = -164.1635206696992

p8a2/min_fs = 0
p8a2/max_fs = 127
p8a2/min_ss = 4224
p8a2/max_ss = 4287
p8a2/fs = -0.001629x +0.999995y
p8a2/ss = -0.999998x -0.001626y
p8a2/corner_x = 397.28135756437644
p8a2/corner_y = -164.2695206696992

p8a3/min_fs = 0
p8a3/max_fs = 127
p8a3/min_ss = 4288
p8a3/max_ss = 4351
p8a3/fs = -0.001629x +0.999995y
p8a3/ss = -0.999998x -0.001626y
p8a3/corner_x = 331.74635756437647
p8a3/corner_y = -164.3765206696992

p8a4/min_fs = 0
p8a4/max_fs = 127
p8a4/min_ss = 4352
p8a4/max_ss = 4415
p8a4/fs = -0.001629x +0.999995y
p8a4/ss = -0.999998x -0.001626y
p8a4/corner_x = 266.2123575643764
p8a4/corner_y = -164.48252066969923

p8a5/min_fs = 0
p8a5/max_fs = 127
p8a5/min_ss = 4416
p8a5/max_ss = 4479
p8a5/fs = -0.001629x +0.999995y
p8a5/ss = -0.999998x -0.001626y
p8a5/corner_x = 200.67735756437648
p8a5/corner_y = -164.58952066969923

p8a6/min_fs = 0
p8a6/max_fs = 127
p8a6/min_ss = 4480
p8a6/max_ss = 4543
p8a6/fs = -0.001629x +0.999995y
p8a6/ss = -0.999998x -0.001626y
p8a6/corner_x = 135.14335756437646
p8a6/corner_y = -164.69552066969922

p8a7/min_fs = 0
p8a7/max_fs = 127
p8a7/min_ss = 4544
p8a7/max_ss = 4607
p8a7/fs = -0.001629x +0.999995y
p8a7/ss = -0.999998x -0.001626y
p8a7/corner_x = 69.60875756437648
p8a7/corner_y = -164.80252066969922

p9a0/min_fs = 0
p9a0/max_fs = 127
p9a0/min_ss = 4608
p9a0/max_ss = 4671
p9a0/fs = -0.000697x +1.000000y
p9a0/ss = -0.999999x -0.000697y
p9a0/corner_x = 526.8163575643764
p9a0/corner_y = -319.36252066969917

p9a1/min_fs = 0
p9a1/max_fs = 127
p9a1/min_ss = 4672
p9a1/max_ss = 4735
p9a1/fs = -0.000697x +1.000000y
p9a1/ss = -0.999999x -0.000697y
p9a1/corner_x = 461.28235756437647
p9a1/corner_y = -319.40752066969924

p9a2/min_fs = 0
p9a2/max_fs = 127
p9a2/min_ss = 4736
p9a2/max_ss = 4799
p9a2/fs = -0.000697x +1.000000y
p9a2/ss = -0.999999x -0.000697y
p9a2/corner_x = 395.74735756437644
p9a2/corner_y = -319.4535206696992

p9a3/min_fs = 0
p9a3/max_fs = 127
p9a3/min_ss = 4800
p9a3/max_ss = 4863
p9a3/fs = -0.000697x +1.000000y
p9a3/ss = -0.999999x -0.000697y
p9a3/corner_x = 330.2123575643764
p9a3/corner_y = -319.4995206696992

p9a4/min_fs = 0
p9a4/max_fs = 127
p9a4/min_ss = 4864
p9a4/max_ss = 4927
p9a4/fs = -0.000697x +1.000000y
p9a4/ss = -0.999999x -0.000697y
p9a4/corner_x = 264.67735756437645
p9a4/corner_y = -319.5445206696992

p9a5/min_fs = 0
p9a5/max_fs = 127
p9a5/min_ss = 4928
p9a5/max_ss = 4991
p9a5/fs = -0.000697x +1.000000y
p9a5/ss = -0.999999x -0.000697y
p9a5/corner_x = 199.14335756437646
p9a5/corner_y = -319.59052066969923

p9a6/min_fs = 0
p9a6/max_fs = 127
p9a6/min_ss = 4992
p9a6/max_ss = 5055
p9a6/fs = -0.000697x +1.000000y
p9a6/ss = -0.999999x -0.000697y
p9a6/corner_x = 133.60835756437646
p9a6/corner_y = -319.63652066969917

p9a7/min_fs = 0
p9a7/max_fs = 127
p9a7/min_ss = 5056
p9a7/max_ss = 5119
p9a7/fs = -0.000697x +1.000000y
p9a7/ss = -0.999999x -0.000697y
p9a7/corner_x = 68.07345756437647
p9a7/corner_y = -319.68152066969924

p10a0/min_fs = 0
p10a0/max_fs = 127
p10a0/min_ss = 5120
p10a0/max_ss = 5183
p10a0/fs = -0.000016x +1.000000y
p10a0/ss = -1.000000x -0.000016y
p10a0/corner_x = 526.7423575643764
p10a0/corner_y = -476.6745206696992

p10a1/min_fs = 0
p10a1/max_fs = 127
p10a1/min_ss = 5184
p10a1/max_ss = 5247
p10a1/fs = -0.000016x +1.000000y
p10a1/ss = -1.000000x -0.000016y
p10a1/corner_x = 461.2073575643764
p10a1/corner_y = -476.67552066969927

p10a2/min_fs = 0
p10a2/max_fs = 127
p10a2/min_ss = 5248
p10a2/max_ss = 5311
p10a2/fs = -0.000016x +1.000000y
p10a2/ss = -1.000000x -0.000016y
p10a2/corner_x = 395.67235756437645
p10a2/corner_y = -476.67652066969924

p10a3/min_fs = 0
p10a3/max_fs = 127
p10a3/min_ss = 5312
p10a3/max_ss = 5375
p10a3/fs = -0.000016x +1.000000y
p10a3/ss = -1.000000x -0.000016y
p10a3/corner_x = 330.13835756437646
p10a3/corner_y = -476.6775206696992

p10a4/min_fs = 0
p10a4/max_fs = 127
p10a4/min_ss = 5376
p10a4/max_ss = 5439
p10a4/fs = -0.000016x +1.000000y
p10a4/ss = -1.000000x -0.000016y
p10a4/corner_x = 264.60335756437644
p10a4/corner_y = -476.6785206696992

p10a5/min_fs = 0
p10a5/max_fs = 127
p10a5/min_ss = 5440
p10a5/max_ss = 5503
p10a5/fs = -0.000016x +1.000000y
p10a5/ss = -1.000000x -0.000016y
p10a5/corner_x = 199.06835756437647
p10a5/corner_y = -476.6795206696992

p10a6/min_fs = 0
p10a6/max_fs = 127
p10a6/min_ss = 5504
p10a6/max_ss = 5567
p10a6/fs = -0.000016x +1.000000y
p10a6/ss = -1.000000x -0.000016y
p10a6/corner_x = 133.53335756437647
p10a6/corner_y = -476.68052066969926

p10a7/min_fs = 0
p10a7/max_fs = 127
p10a7/min_ss = 5568
p10a7/max_ss = 5631
p10a7/fs = -0.000016x +1.000000y
p10a7/ss = -1.000000x -0.000016y
p10a7/corner_x = 67.99955756437647
p10a7/corner_y = -476.68152066969924

p11a0/min_fs = 0
p11a0/max_fs = 127
p11a0/min_ss = 5632
p11a0/max_ss = 5695
p11a0/fs = -0.001178x +0.999998y
p11a0/ss = -0.999999x -0.001179y
p11a0/corner_x = 529.3103575643764
p11a0/corner_y = -637.0765206696992

p11a1/min_fs = 0
p11a1/max_fs = 127
p11a1/min_ss = 5696
p11a1/max_ss = 5759
p11a1/fs = -0.001178x +0.999998y
p11a1/ss = -0.999999x -0.001179y
p11a1/corner_x = 463.77635756437644
p11a1/corner_y = -637.1545206696992

p11a2/min_fs = 0
p11a2/max_fs = 127
p11a2/min_ss = 5760
p11a2/max_ss = 5823
p11a2/fs = -0.001178x +0.999998y
p11a2/ss = -0.999999x -0.001179y
p11a2/corner_x = 398.2413575643764
p11a2/corner_y = -637.2315206696992

p11a3/min_fs = 0
p11a3/max_fs = 127
p11a3/min_ss = 5824
p11a3/max_ss = 5887
p11a3/fs = -0.001178x +0.999998y
p11a3/ss = -0.999999x -0.001179y
p11a3/corner_x = 332.70635756437645
p11a3/corner_y = -637.3085206696992

p11a4/min_fs = 0
p11a4/max_fs = 127
p11a4/min_ss = 5888
p11a4/max_ss = 5951
p11a4/fs = -0.001178x +0.999998y
p11a4/ss = -0.999999x -0.001179y
p11a4/corner_x = 267.1713575643764
p11a4/corner_y = -637.3855206696992

p11a5/min_fs = 0
p11a5/max_fs = 127
p11a5/min_ss = 5952
p11a5/max_ss = 6015
p11a5/fs = -0.001178x +0.999998y
p11a5/ss = -0.999999x -0.001179y
p11a5/corner_x = 201.63635756437648
p11a5/corner_y = -637.4635206696993

p11a6/min_fs = 0
p11a6/max_fs = 127
p11a6/min_ss = 6016
p11a6/max_ss = 6079
p11a6/fs = -0.001178x +0.999998y
p11a6/ss = -0.999999x -0.001179y
p11a6/corner_x = 136.10235756437646
p11a6/corner_y = -637.5405206696993

p11a7/min_fs = 0
p11a7/max_fs = 127
p11a7/min_ss = 6080
p11a7/max_ss = 6143
p11a7/fs = -0.001178x +0.999998y
p11a7/ss = -0.999999x -0.001179y
p11a7/corner_x = 70.56775756437646
p11a7/corner_y = -637.6175206696993

p12a0/min_fs = 0
p12a0/max_fs = 127
p12a0/min_ss = 6144
p12a0/max_ss = 6207
p12a0/fs = -0.000343x +1.000000y
p12a0/ss = -1.000000x -0.000343y
p12a0/corner_x = 534.1580454931367
p12a0/corner_y = 489.98302795528906

p12a1/min_fs = 0
p12a1/max_fs = 127
p12a1/min_ss = 6208
p12a1/max_ss = 6271
p12a1/fs = -0.000343x +1.000000y
p12a1/ss = -1.000000x -0.000343y
p12a1/corner_x = 468.6230454931367
p12a1/corner_y = 489.961027955289

p12a2/min_fs = 0
p12a2/max_fs = 127
p12a2/min_ss = 6272
p12a2/max_ss = 6335
p12a2/fs = -0.000343x +1.000000y
p12a2/ss = -1.000000x -0.000343y
p12a2/corner_x = 403.08804549313675
p12a2/corner_y = 489.93802795528904

p12a3/min_fs = 0
p12a3/max_fs = 127
p12a3/min_ss = 6336
p12a3/max_ss = 6399
p12a3/fs = -0.000343x +1.000000y
p12a3/ss = -1.000000x -0.000343y
p12a3/corner_x = 337.5530454931367
p12a3/corner_y = 489.91602795528905

p12a4/min_fs = 0
p12a4/max_fs = 127
p12a4/min_ss = 6400
p12a4/max_ss = 6463
p12a4/fs = -0.000343x +1.000000y
p12a4/ss = -1.000000x -0.000343y
p12a4/corner_x = 272.01904549313673
p12a4/corner_y = 489.893027955289

p12a5/min_fs = 0
p12a5/max_fs = 127
p12a5/min_ss = 6464
p12a5/max_ss = 6527
p12a5/fs = -0.000343x +1.000000y
p12a5/ss = -1.000000x -0.000343y
p12a5/corner_x = 206.48404549313673
p12a5/corner_y = 489.87102795528904

p12a6/min_fs = 0
p12a6/max_fs = 127
p12a6/min_ss = 6528
p12a6/max_ss = 6591
p12a6/fs = -0.000343x +1.000000y
p12a6/ss = -1.000000x -0.000343y
p12a6/corner_x = 140.94904549313674
p12a6/corner_y = 489.848027955289

p12a7/min_fs = 0
p12a7/max_fs = 127
p12a7/min_ss = 6592
p12a7/max_ss = 6655
p12a7/fs = -0.000343x +1.000000y
p12a7/ss = -1.000000x -0.000343y
p12a7/corner_x = 75.41424549313672
p12a7/corner_y = 489.826027955289

p13a0/min_fs = 0
p13a0/max_fs = 127
p13a0/min_ss = 6656
p13a0/max_ss = 6719
p13a0/fs = +0.000326x +1.000000y
p13a0/ss = -0.999999x +0.000326y
p13a0/corner_x = 533.9060454931367
p13a0/corner_y = 330.052027955289

p13a1/min_fs = 0
p13a1/max_fs = 127
p13a1/min_ss = 6720
p13a1/max_ss = 6783
p13a1/fs = +0.000326x +1.000000y
p13a1/ss = -0.999999x +0.000326y
p13a1/corner_x = 468.3710454931367
p13a1/corner_y = 330.074027955289

p13a2/min_fs = 0
p13a2/max_fs = 127
p13a2/min_ss = 6784
p13a2/max_ss = 6847
p13a2/fs = +0.000326x +1.000000y
p13a2/ss = -0.999999x +0.000326y
p13a2/corner_x = 402.8370454931367
p13a2/corner_y = 330.095027955289

p13a3/min_fs = 0
p13a3/max_fs = 127
p13a3/min_ss = 6848
p13a3/max_ss = 6911
p13a3/fs = +0.000326x +1.000000y
p13a3/ss = -0.999999x +0.000326y
p13a3/corner_x = 337.30204549313675
p13a3/corner_y = 330.11602795528904

p13a4/min_fs = 0
p13a4/max_fs = 127
p13a4/min_ss = 6912
p13a4/max_ss = 6975
p13a4/fs = +0.000326x +1.000000y
p13a4/ss = -0.999999x +0.000326y
p13a4/corner_x = 271.7670454931367
p13a4/corner_y = 330.13802795528903

p13a5/min_fs = 0
p13a5/max_fs = 127
p13a5/min_ss = 6976
p13a5/max_ss = 7039
p13a5/fs = +0.000326x +1.000000y
p13a5/ss = -0.999999x +0.000326y
p13a5/corner_x = 206.23204549313672
p13a5/corner_y = 330.15902795528905

p13a6/min_fs = 0
p13a6/max_fs = 127
p13a6/min_ss = 7040
p13a6/max_ss = 7103
p13a6/fs = +0.000326x +1.000000y
p13a6/ss = -0.999999x +0.000326y
p13a6/corner_x = 140.69804549313673
p13a6/corner_y = 330.18102795528904

p13a7/min_fs = 0
p13a7/max_fs = 127
p13a7/min_ss = 7104
p13a7/max_ss = 7167
p13a7/fs = +0.000326x +1.000000y
p13a7/ss = -0.999999x +0.000326y
p13a7/corner_x = 75.16344549313672
p13a7/corner_y = 330.20202795528905

p14a0/min_fs = 0
p14a0/max_fs = 127
p14a0/min_ss = 7168
p14a0/max_ss = 7231
p14a0/fs = +0.000486x +1.000000y
p14a0/ss = -1.000000x +0.000486y
p14a0/corner_x = 534.0600454931367
p14a0/corner_y = 173.41302795528907

p14a1/min_fs = 0
p14a1/max_fs = 127
p14a1/min_ss = 7232
p14a1/max_ss = 7295
p14a1/fs = +0.000486x +1.000000y
p14a1/ss = -1.000000x +0.000486y
p14a1/corner_x = 468.5250454931367
p14a1/corner_y = 173.44502795528908

p14a2/min_fs = 0
p14a2/max_fs = 127
p14a2/min_ss = 7296
p14a2/max_ss = 7359
p14a2/fs = +0.000486x +1.000000y
p14a2/ss = -1.000000x +0.000486y
p14a2/corner_x = 402.99004549313673
p14a2/corner_y = 173.47702795528906

p14a3/min_fs = 0
p14a3/max_fs = 127
p14a3/min_ss = 7360
p14a3/max_ss = 7423
p14a3/fs = +0.000486x +1.000000y
p14a3/ss = -1.000000x +0.000486y
p14a3/corner_x = 337.4550454931367
p14a3/corner_y = 173.50902795528907

p14a4/min_fs = 0
p14a4/max_fs = 127
p14a4/min_ss = 7424
p14a4/max_ss = 7487
p14a4/fs = +0.000486x +1.000000y
p14a4/ss = -1.000000x +0.000486y
p14a4/corner_x = 271.9210454931367
p14a4/corner_y = 173.54102795528908

p14a5/min_fs = 0
p14a5/max_fs = 127
p14a5/min_ss = 7488
p14a5/max_ss = 7551
p14a5/fs = +0.000486x +1.000000y
p14a5/ss = -1.000000x +0.000486y
p14a5/corner_x = 206.38604549313672
p14a5/corner_y = 173.57302795528906

p14a6/min_fs = 0
p14a6/max_fs = 127
p14a6/min_ss = 7552
p14a6/max_ss = 7615
p14a6/fs = +0.000486x +1.000000y
p14a6/ss = -1.000000x +0.000486y
p14a6/corner_x = 140.85104549313672
p14a6/corner_y = 173.60502795528907

p14a7/min_fs = 0
p14a7/max_fs = 127
p14a7/min_ss = 7616
p14a7/max_ss = 7679
p14a7/fs = +0.000486x +1.000000y
p14a7/ss = -1.000000x +0.000486y
p14a7/corner_x = 75.31634549313671
p14a7/corner_y = 173.63602795528908

p15a0/min_fs = 0
p15a0/max_fs = 127
p15a0/min_ss = 7680
p15a0/max_ss = 7743
p15a0/fs = -0.000121x +0.999999y
p15a0/ss = -1.000000x -0.000121y
p15a0/corner_x = 533.0570454931367
p15a0/corner_y = 17.89922795528906

p15a1/min_fs = 0
p15a1/max_fs = 127
p15a1/min_ss = 7744
p15a1/max_ss = 7807
p15a1/fs = -0.000121x +0.999999y
p15a1/ss = -1.000000x -0.000121y
p15a1/corner_x = 467.5220454931367
p15a1/corner_y = 17.89132795528906

p15a2/min_fs = 0
p15a2/max_fs = 127
p15a2/min_ss = 7808
p15a2/max_ss = 7871
p15a2/fs = -0.000121x +0.999999y
p15a2/ss = -1.000000x -0.000121y
p15a2/corner_x = 401.9880454931367
p15a2/corner_y = 17.883427955289058

p15a3/min_fs = 0
p15a3/max_fs = 127
p15a3/min_ss = 7872
p15a3/max_ss = 7935
p15a3/fs = -0.000121x +0.999999y
p15a3/ss = -1.000000x -0.000121y
p15a3/corner_x = 336.4530454931367
p15a3/corner_y = 17.875427955289062

p15a4/min_fs = 0
p15a4/max_fs = 127
p15a4/min_ss = 7936
p15a4/max_ss = 7999
p15a4/fs = -0.000121x +0.999999y
p15a4/ss = -1.000000x -0.000121y
p15a4/corner_x = 270.91804549313673
p15a4/corner_y = 17.86752795528906

p15a5/min_fs = 0
p15a5/max_fs = 127
p15a5/min_ss = 8000
p15a5/max_ss = 8063
p15a5/fs = -0.000121x +0.999999y
p15a5/ss = -1.000000x -0.000121y
p15a5/corner_x = 205.38304549313673
p15a5/corner_y = 17.85952795528906

p15a6/min_fs = 0
p15a6/max_fs = 127
p15a6/min_ss = 8064
p15a6/max_ss = 8127
p15a6/fs = -0.000121x +0.999999y
p15a6/ss = -1.000000x -0.000121y
p15a6/corner_x = 139.8490454931367
p15a6/corner_y = 17.85162795528906

p15a7/min_fs = 0
p15a7/max_fs = 127
p15a7/min_ss = 8128
p15a7/max_ss = 8191
p15a7/fs = -0.000121x +0.999999y
p15a7/ss = -1.000000x -0.000121y
p15a7/corner_x = 74.31384549313671
p15a7/corner_y = 17.843627955289058
