; KA: Updated geometry to match det_2160_lowq7.h5
; KA: Keeping only 8 low-Q panels
; KA: Manually optimized with hdfsee to match run 48 AgBe rings
; KA: Manually optimized with hdfsee to match AgBe rings (24.05.2019)
; Manually optimized with hdfsee
; Manually optimized with hdfsee
; Optimized panel offsets can be found at the end of the file
; OY: I think this is a very well optimized geometry!
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Camera length from LiTiO calibration
; Manually optimized with hdfsee
; Now all distances between panels is 5.8mm (29 pixels)
; OY: ACHTUNG! Orientation of the 2 halves of the detector might be wrong!
; A bit changed by OY: now 128 panels, rigid groups, bad lines to mask double pixels.
; Fixed errors noticed by Oleksandr
; Beginning of an AGIPD geometry construction by Helen Ginn on Thursday before beam time.
; Global panel positions not guesses
; Local positioning largely - but not entirely - guesses
; fast and slow scan directions have the potential to be accurate.

adu_per_eV = 0.0075  ; 45 ADU per 6000 eV
clen = 0.120
photon_energy = 12000
res = 5000 ; 200 um pixels

dim0 = %
dim1 = ss
dim2 = fs
data = /data/data

mask = /entry_1/instrument_1/detector_1/mask
mask_good = 0x0000
mask_bad = 0xfeff

rigid_group_q0 = p0a0,p0a1
rigid_group_q1 = p1a0,p1a1
rigid_group_q2 = p2a0,p2a1
rigid_group_q3 = p3a0,p3a1

rigid_group_p0 = p0a0,p0a1
rigid_group_p1 = p1a0,p1a1
rigid_group_p2 = p2a0,p2a1
rigid_group_p3 = p3a0,p3a1

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15

p0a0/min_fs = 0
p0a0/min_ss = 0
p0a0/max_fs = 127
p0a0/max_ss = 63
p0a0/fs = +0.001590x -0.999999y
p0a0/ss = +0.999999x +0.001590y
p0a0/corner_x = -127.083
p0a0/corner_y = 187.647

p0a1/min_fs = 0
p0a1/min_ss = 64
p0a1/max_fs = 127
p0a1/max_ss = 127
p0a1/fs = +0.001590x -0.999999y
p0a1/ss = +0.999999x +0.001590y
p0a1/corner_x = -61.8079
p0a1/corner_y = 188.142

p1a0/min_fs = 0
p1a0/min_ss = 128
p1a0/max_fs = 127
p1a0/max_ss = 191
p1a0/fs = -0.001899x -0.999999y
p1a0/ss = +0.999999x -0.001899y
p1a0/corner_x = -139.702
p1a0/corner_y = 19.5714

p1a1/min_fs = 0
p1a1/min_ss = 192
p1a1/max_fs = 127
p1a1/max_ss = 255
p1a1/fs = -0.001899x -0.999999y
p1a1/ss = +0.999999x -0.001899y
p1a1/corner_x = -75.3187
p1a1/corner_y = 19.4941

p2a0/min_fs = 0
p2a0/min_ss = 256
p2a0/max_fs = 127
p2a0/max_ss = 319
p2a0/fs = +0.000124x +1.000000y
p2a0/ss = -1.000000x +0.000124y
p2a0/corner_x = 149.178
p2a0/corner_y = -142.563

p2a1/min_fs = 0
p2a1/min_ss = 320
p2a1/max_fs = 127
p2a1/max_ss = 383
p2a1/fs = +0.000124x +1.000000y
p2a1/ss = -1.000000x +0.000124y
p2a1/corner_x = 83.9773
p2a1/corner_y = -143.006

p3a0/min_fs = 0
p3a0/min_ss = 384
p3a0/max_fs = 127
p3a0/max_ss = 447
p3a0/fs = +0.001282x +1.000001y
p3a0/ss = -1.000001x +0.001282y
p3a0/corner_x = 155.666
p3a0/corner_y = 45.9387

p3a1/min_fs = 0
p3a1/min_ss = 448
p3a1/max_fs = 127
p3a1/max_ss = 511
p3a1/fs = +0.001282x +1.000001y
p3a1/ss = -1.000001x +0.001282y
p3a1/corner_x = 90.479
p3a1/corner_y = 46.0414
