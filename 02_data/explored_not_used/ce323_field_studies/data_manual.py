"""Values entered by hand from PDF text / images, each with its provenance.
Every number here was read from the source named alongside it and cross-checked
(row/column totals, text layer vs image) where the source allowed.
"""

# ---------------------------------------------------------------- Group 12, Exp 2 (V-Box) -- CE323_... VBOX_Group12.pdf p.2-3 (text layer)
G12_E2_SPEED_PCTL = {15: 42.307, 25: 46.86675, 50: 53.7105, 85: 59.2188, 95: 60.8225}
G12_E2_ACC_PCTL = {  # percentile: (long_acc, long_dec, lat_acc)  m/s^2
    15: (0.11, 0.11, 0.11), 25: (0.18, 0.18, 0.18), 50: (0.37, 0.38, 0.38),
    85: (0.82, 0.87, 0.85), 95: (1.17, 1.31, 1.244)}
G12_E2_KS = {'normal': 4361.35, 'weibull': 3146.912, 'lognormal': 7538.90}

# ---------------------------------------------------------------- Group 12, Exp 3 -- exp 3.pdf (text layer)
G12_E3_LANE_2MIN = {  # lane: [(interval, vehicles, pcu)]
    'Inner': [('0-2', 32, 50.1), ('2-4', 17, 30.3), ('4-6', 18, 36.7), ('6-8', 20, 44.0), ('8-10', 19, 37.6)],
    'Outer': [('0-2', 28, 38.7), ('2-4', 29, 32.3), ('4-6', 25, 17.7), ('6-8', 29, 32.4), ('8-10', 30, 40.8)],
    'Combined': [('0-2', 60, 88.8), ('2-4', 46, 62.6), ('4-6', 43, 54.4), ('6-8', 49, 76.4), ('8-10', 49, 78.4)],
}
G12_E3_FFS = {  # class: (avg, p50, p85)  km/h  -- "Free Flow Speed Statistics"
    'Auto': (24.41, 24.59, 27.67), 'Bus': (36.68, 32.67, 44.72), 'Car': (41.88, 41.53, 48.62),
    'LCV': (39.19, 37.93, 47.90), 'Multi Axle': (23.89, 22.62, 27.69), 'Truck': (35.35, 35.30, 41.34),
    'Two Wheeler': (45.04, 40.51, 47.81)}
G12_E3_STREAM = {  # class: (avg, p85, median) -- "Combined Speed Statistics"
    'Car': (41.88, 48.62, 41.53), 'Truck': (35.35, 41.34, 35.30), 'LCV': (39.19, 47.90, 37.93),
    'Two Wheeler': (45.04, 47.81, 40.51)}
G12_E3_NORMALITY = {'mean': 40.71, 'sd': 18.71, 'dof': 10, 'chi2_crit': 18.31, 'chi2_calc': None,
                    'conclusion': 'Reject Normal Distribution'}

# ---------------------------------------------------------------- Group 12, Exp 4 -- Group12_exp4.pdf
G12_E4_SUMMARY = {  # text layer p.3
    'Lane 1 (Inner)': dict(n=113, mean=5.334, sd=2.734, min=0.042),
    'Lane 2 (Outer)': dict(n=139, mean=4.429, sd=2.500, min=0.306),
    'Combined': dict(n=252, mean=2.437, sd=1.783, min=0.042)}
G12_E4_PCTL = {5: 0.076, 15: 1.00, 50: 7.50, 85: 402.15, 95: 527.83}  # text layer p.4 (as printed)
# Frequency tables = embedded images (p.3-4), read at native resolution; totals reconcile (see validation)
_bins = [(round(i * 0.5, 1), round(i * 0.5 + 0.5, 1)) for i in range(40)]
G12_E4_BINS = {
    'Lane 1 (Inner)': [5, 6, 2, 4, 5, 3, 4, 4, 7, 12, 2, 6, 13, 4, 7, 5, 10, 6, 2, 3, 1, 0, 0, 0, 0, 1],
    'Lane 2 (Outer)': [7, 2, 5, 6, 6, 8, 12, 6, 13, 18, 16, 15, 12, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'Combined': [41, 27, 19, 28, 24, 21, 19, 17, 22, 14, 4, 8, 3, 2, 1, 0, 0, 0, 0, 0, 1]}
G12_E4_GOF = [  # image p.8 'Result Summary'
    ('Lane 1', 'normal', 5.333535714, 2.734333342, 25.89529843, 21.02606982, 0.011105972, 12, 'Not Appropriate'),
    ('Lane 1', 'exponential', 5.333535714, 2.734333342, 86.52306066, 22.36203249, 0, 13, 'Not Appropriate'),
    ('Lane 2', 'normal', 4.429282609, 2.50045334, 27.5051293, 22.36203249, 0.010601231, 13, 'Not Appropriate'),
    ('Lane 2', 'exponential', 4.429282609, 2.50045334, 146.7987247, 24.99579014, 0, 15, 'Not Appropriate'),
    ('Combined', 'normal', 2.436729084, 1.783470717, 16.92744022, 18.30703805, 0.075985041, 10, 'Appropriate'),
    ('Combined', 'exponential', 2.436729084, 1.783470717, 52.76472922, 22.36203249, 9.92956e-07, 13, 'Not Appropriate')]

# ---------------------------------------------------------------- Group 12, Exp 5 -- CE323_EXP5.pdf observation table (p.4 via pdfplumber; p.5 via word x-positions, verified against page image)
G12_E5_P5 = {
    ('Truck', 'Free'): [42],
    ('Truck', 'Forced'): [28, 31, 44, 32, 44, 43, 28, 25, 29, 31],
    ('Bus', 'Forced'): [37, 20, 39, 37, 37, 35, 31, 34, 43, 56, 45, 46, 41, 40, 32, 34, 48, 46, 49, 44, 39, 56],
    ('LCV', 'Forced'): [28, 44, 43, 38, 45, 41, 37, 33, 35, 36, 38, 43, 32, 33, 35, 20, 24, 42, 39, 31, 33, 29],
    ('Car', 'Forced'): [29, 47, 27, 32, 43, 46, 40, 32, 34, 49, 51, 61, 65, 53, 29, 25, 24, 21, 50, 50, 34, 36, 39, 56, 54],
    ('Two Wheeler', 'Forced'): [49, 52, 31, 51, 49, 50, 35, 24, 50, 52]}
G12_E5_REPORTED = {  # 'Comparison' table p.9 (text): class: (N_free, N_comb, mean_f, mean_c, p85_f, p85_c, p50_f, p50_c, sd_f, sd_c)
    'Truck': (49, 49, 35.61, 37, 42, 42.8, 36, 36, 5.3, 6.96),
    'LCV': (55, 55, 40.21, 46.14, 54.25, 47.9, 46, 40, 8.47, 8.64),
    'Car': (58, 59, 44.81, 57.4, 64.7, 56, 55, 46, 13.72, 14.05),
    'Two Wheeler': (44, 44, 46.56, 45.07, 64.8, 60.1, 47, 48.5, 19.48, 14.17)}

# ---------------------------------------------------------------- Group 12, Exp 6 -- CE323_EXP6.pdf p.4-5 (pdfplumber tables)
G12_E6_CLASSES = ['Multi Axle', 'Truck', 'LCV', 'Bus', 'Car', 'Auto', 'Two Wheeler']
G12_E6_LAPS = [  # (from_min, to_min, total_min, overtaking[7], overtaken[7], pcu_overtaking, pcu_overtaken)
    (0, 5.34, 5.34, [0, 0, 2, 0, 5, 0, 8], [0, 1, 0, 0, 0, 1, 1], 11.8, 3.9),
    (5.34, 10.47, 5.13, [0, 0, 5, 1, 21, 4, 34], [0, 0, 1, 0, 3, 2, 4], 52, 8.8),
    (10.47, 22.14, 11.27, [0, 0, 0, 0, 1, 0, 10], [0, 0, 3, 0, 3, 5, 7], 6, 16.7),
    (22.14, 26.58, 4.44, [0, 0, 0, 0, 1, 1, 5], [0, 0, 0, 0, 0, 5, 6], 4.7, 9)]
G12_E6_OPPOSITE = [  # MA, Truck, LCV, Bus, Car, Auto, 2W, Cycle, PCU
    [2, 14, 26, 10, 46, 10, 40, 2, 217.2], [5, 36, 77, 6, 110, 22, 134, 1, 521.8],
    [5, 3, 34, 12, 136, 20, 73, 2, 339.8], [0, 32, 53, 4, 85, 25, 107, 6, 396.1]]
G12_E6_RESULT = [(1175.8427, 0.089, 21.34831461), (2798.8304, 0.0855, 22.22222222),
                 (933.00799, 0.18783333, 10.11535049), (2705.4054, 0.074, 25.67567568)]
G12_E6_AVG = (1903.2716, 0.10908333, 19.84039075)
G12_E6_L_KM = 1.9

# ---------------------------------------------------------------- Group 12, Exp 7 -- CE323_EXP7 (1).pdf p.2 (text layer)
G12_E7_COUNTS = {  # (route, movement): [Truck, LCV, Bus, Car, Auto, 2W]
    ('Route 1', 'Straight'): [17, 29, 8, 87, 12, 87], ('Route 1', 'Right turn'): [0, 4, 0, 21, 5, 18],
    ('Route 2', 'Straight'): [12, 34, 14, 111, 13, 129], ('Route 2', 'Left turn'): [1, 0, 0, 6, 2, 12],
    ('Route 3', 'Right turn'): [0, 0, 0, 3, 2, 6], ('Route 3', 'Left turn'): [0, 3, 0, 18, 8, 11]}
G12_E7_PCU = {'Truck': 3.0, 'LCV': 1.5, 'Bus': 3.0, 'Car': 1.0, 'Auto': 1.0, 'Two Wheeler': 0.5}
G12_E7_REPORTED = dict(lane_pcu={('Route 1', 'Straight'): 261, ('Route 1', 'Right turn'): 41, ('Route 2', 'Straight'): 317.5,
                                 ('Route 2', 'Left turn'): 17, ('Route 3', 'Right turn'): 8, ('Route 3', 'Left turn'): 36},
                       crit=[261, 317.5, 36], crit_hr=[783, 952.5, 108], sat_flow=1800, sat_h=2, vs=[0.435, 0.528, 0.06],
                       sum_vs=1.023, sum_vs_used=0.9, L=6, C0=140, green=[58, 70, 8], amber=[2, 2, 2], red=[80, 68, 130])

# ---------------------------------------------------------------- Group 12, Exp 8 -- CE323_EXP8.pdf p.3-4 (text layer). Rows = bays.
G12_E8_TIMES = ['10:40', '10:50', '11:00', '11:10', '11:20', '11:30', '11:40', '11:50', '12:00']
G12_E8_ROWS = [  # (type, [plate-last-4 per snapshot, '0' = empty], reported duration min)
    ('bike', ['0', '0', '0', '0', '0', '0', '0', '2649', '2649'], 20),
    ('bike', ['3714'] * 9, 90),
    ('bike', ['7230'] * 5 + ['0'] * 4, 50),
    ('bike', ['0', '0', '0', '0', '1824', '0', '0', '0', '0'], 10),
    ('bike', ['0', '0', '0', '4611', '4611', '0', '0', '0', '0'], 20),
    ('bike', ['3415'] * 9, 90),
    ('bike', ['1583'] * 9, 90),
    ('bike', ['2001', '2001', '2001', '4837', '4837', '4837', '4837', '9739', '9739'], 90),
    ('bike', ['7375'] * 7 + ['0'] * 2, 70),
    ('bike', ['5221'] * 7 + ['0'] * 2, 70),
    ('bike', ['9347'] * 9, 90),
    ('bike', ['1465'] * 9, 90),
    ('bike', ['4529'] * 9, 90),
    ('bike', ['7230', '7230'] + ['2754'] * 7, 90),
    ('bike', ['5221', '1583'] + ['7800'] * 7, 90),
    ('bike', ['2388'] * 6 + ['0'] * 3, 60),
    ('bike', ['2754'] * 2 + ['0'] * 7, 20),
    ('bike', ['0237'] * 5 + ['0'] * 4, 50),
    ('bike', ['3714'] * 2 + ['0'] * 7, 20),
    ('bike', ['4298'] * 9, 90),
    ('bike', ['9509'] * 9, 90),
    ('bike', ['3409'] * 9, 90),
    ('bike', ['2632'] * 2 + ['0'] * 7, 20),
    ('bike', ['3747'] * 9, 90),
    ('bike', ['3084'] * 9, 90),
    ('bike', ['2649'] * 5 + ['0'] * 4, 50),
    ('bike', ['6185'] * 9, 90),
    ('bike', ['6095'] * 9, 90),
    ('bike', ['6873'] * 9, 90),
    ('bike', ['9092'] * 3 + ['0'] * 6, 30),
    ('bike', ['1030'] * 9, 90),
    ('bike', ['7848'] * 9, 90),
    ('bike', ['0'] * 5 + ['9986'] * 4, 40),
    ('bike', ['0'] * 7 + ['5221'] * 2, 20),
    ('bike', ['0'] * 7 + ['9296'] * 2, 20),
    ('bike', ['0', '0'] + ['9509'] * 7, 70),
    ('bike', ['0', '0'] + ['2332'] * 3 + ['0'] * 4, 30),
    ('bike', ['0', '0'] + ['5762'] * 7, 70),
    ('car', ['3519'] * 9, 90), ('car', ['9396'] * 9, 90), ('car', ['7135'] * 9, 90), ('car', ['1271'] * 9, 90),
    ('car', ['7248'] * 9, 90), ('car', ['1913'] * 9, 90), ('car', ['6500'] * 9, 90),
    ('car', ['0842'] * 7 + ['0'] * 2, 70),
    ('car', ['4334'] * 9, 90), ('car', ['6643'] * 9, 90), ('car', ['2185'] * 9, 90), ('car', ['2428'] * 9, 90),
    ('car', ['0'] + ['9965'] * 8, 80),
    ('car', ['0', '0'] + ['0775K'] * 7, 70)]
G12_E8_ROW_TOTALS = [44, 45, 46, 46, 46, 41, 40, 40, 40]  # printed 'total no of vehicles' row
G12_E8_REPORTED = dict(accum_table=[41, 43, 44, 44, 44, 39, 38, 38, 38], max_accum_text=46, max_accum_result=44,
                       avg_accum=41, load_veh_min=3690, parking_index_pct=78.85, index_capacity_text=49,
                       index_capacity_used=52, volume=52, avg_duration=70.38, car_n=14, car_avg=91.42,
                       bike_n=38, bike_avg=64.47, arrivals=[4, 1, 4, 1, 1, 0, 0, 3, 0],
                       departures=[2, 0, 3, 1, 0, 6, 1, 3, 0], arr_rate=10, dep_rate=10)

# ---------------------------------------------------------------- Group 12, Exp 9 -- Experiment9_grp 12.pdf
G12_E9_EF = {'Walk': (33, 9, 3.67), 'Cycle': (137, 18, 7.61), 'Two Wheeler': (205, 20, 10.25), 'Car': (60, 6, 10.00)}
# partial O-D matrices (embedded images p.3-4). {(origin, dest): trips}
G12_E9_PARTIAL = {
    'Walk': {(1, 10): 1, (4, 9): 1, (4, 10): 1, (4, 11): 1, (10, 1): 1, (10, 4): 3, (11, 4): 1},
    'Cycle': {(1, 4): 1, (1, 9): 1, (2, 10): 1, (3, 9): 1, (4, 10): 3, (4, 11): 3, (9, 1): 1, (9, 3): 1,
              (10, 2): 1, (10, 4): 1, (11, 4): 4},
    'Two Wheeler': {(1, 2): 1, (1, 10): 1, (1, 11): 1, (2, 11): 1, (3, 10): 1, (4, 7): 1, (4, 9): 2, (4, 10): 1,
                    (4, 12): 1, (7, 4): 1, (9, 4): 4, (10, 1): 1, (10, 4): 1, (11, 2): 1, (11, 3): 1, (12, 4): 1},
    'Car': {(4, 6): 2, (5, 8): 1, (6, 4): 2, (8, 6): 1}}
G12_E9_PARTIAL_TOTALS = {'Walk': 9, 'Cycle': 18, 'Two Wheeler': 20, 'Car': 6}
# handwritten 'Traffic Volume Count' sheet (image p.9); 15-min intervals. First row written as sums (4+1+1, 12+12, 18+16, 3+?+7)
G12_E9_VOLUME = {'Walk': [6, 4, 7, 3, 5, 8], 'Cycle': [24, 18, 28, 15, 20, 32],
                 'Two Wheeler': [34, 28, 42, 22, 31, 48], 'Car': [10, 8, 12, 7, 9, 14]}
G12_E9_SITE = dict(lat=26.187478, lon=91.689866, photo_ts='2026-03-23 10:33 / 11:46 IST (GPS Map Camera)')

# ---------------------------------------------------------------- Group 5, Exp 2 (V-Box) -- Transportation Lab 2 (1).docx tables
G5_E2 = [
    ('Total data entries', 4961, '-'), ('Valid observations (speed>0)', 3994, '-'),
    ('Speed min', 0.632, 'km/h'), ('Speed max', 67.342, 'km/h'), ('Speed mean', 55.51, 'km/h'),
    ('Speed p5 (labelled p15 in Results table)', 46.96, 'km/h'), ('Speed p50', 58.77, 'km/h'),
    ('Speed p85', 63.44, 'km/h'), ('Speed p95', 65.60, 'km/h'), ('Speed SD', 10.1322, 'km/h'),
    ('Accel min', 0.01, 'm/s2'), ('Accel max', 0.95, 'm/s2'), ('Accel mean', 0.1604, 'm/s2'), ('Accel p15', 0.04, 'm/s2'),
    ('Accel p25', 0.06, 'm/s2'), ('Accel p50', 0.11, 'm/s2'), ('Accel p85', 0.25, 'm/s2'), ('Accel p95', 0.63, 'm/s2'),
    ('Accel variance', 0.02873, '(m/s2)^2'),
    ('Decel min', 0.01, 'm/s2'), ('Decel max', 2.07, 'm/s2'), ('Decel mean', 0.3002, 'm/s2'), ('Decel p15', 0.04, 'm/s2'),
    ('Decel p25', 0.06, 'm/s2'), ('Decel p50', 0.17, 'm/s2'), ('Decel p85', 0.46, 'm/s2'), ('Decel p95', 1.266, 'm/s2'),
    ('Decel variance', 0.15465, '(m/s2)^2'),
    ('LatAcc min', -1.17, 'm/s2'), ('LatAcc max', 0.94, 'm/s2'), ('LatAcc mean', 0.0182, 'm/s2'), ('LatAcc p15', -0.19, 'm/s2'),
    ('LatAcc p25', -0.08, 'm/s2'), ('LatAcc p50', 0.00, 'm/s2'), ('LatAcc p85', 0.28, 'm/s2'), ('LatAcc p95', 0.48, 'm/s2'),
    ('LatAcc SD', 0.2694, 'm/s2'), ('Best-fit speed distribution', 'Weibull (min chi-square)', '-')]

# ---------------------------------------------------------------- Individual (Ayush Kumar), Exp 3 -- pie-chart labels p.6 (read at 200 dpi)
AK_E3_COMPOSITION_PCT = {
    'Lane 2': {'Truck': 1.840490798, 'LCV': 4.294478528, 'Bus': 7.36196319, 'Car': 23.92638037, 'Auto': 17.17791411, 'Two Wheeler': 45.39877301},
    'Lane 1': {'Truck': 9.836065574, 'LCV': 18.03278689, 'Bus': 5.737704918, 'Car': 56.55737705, 'Auto': 0.819672131, 'Two Wheeler': 9.016393443}}
AK_E5_COUNTS = {'Truck': 38, 'LCV': 45, 'Bus': 37, 'Car': 102, 'Auto': 34, 'Two Wheeler': 89}
AK_E5_RESULTS = [('Car mean speed', '≈54.6'), ('Two Wheeler mean speed', '≈52'), ('Auto mean speed', '≈33'), ('Car p85 speed', '≈67')]
