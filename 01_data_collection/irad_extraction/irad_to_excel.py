#!/usr/bin/env python3
"""
iRAD / eDAR accident-report PDFs  ->  one structured Excel workbook (+ optional Parquet/CSV)

Reads every "Accident Summary" PDF under a folder (recursively), parses the bordered
label/value tables section by section, and writes one sheet per entity:

    Accidents, Vehicles, Drivers, Passengers, Pedestrians, Witnesses,
    Transport_Inspections, Road_Details, Hospital_Details, TMS_Details,
    Validation_Issues, Files_Log, Data_Dictionary, Summary

Principles ("no false entries"):
  * Every value comes verbatim from the PDF (only whitespace/newlines normalised).
  * Only the literal "no data" tokens  NA, N/A, empty  are turned into blank cells.
    Meaningful categories such as "Not Known" / "Not Applicable" are kept as-is.
  * Typed helper columns (real dates, numbers, lat/lon) are ADDED next to the raw
    text; if a value can't be parsed with an explicit format, the helper stays blank
    and the problem is logged. Nothing is guessed or imputed.
  * Inconsistencies (totals that don't add up, coordinates outside Assam, severity
    vs. deaths, duplicates, unknown labels, ...) are LOGGED in Validation_Issues,
    never "fixed" silently.
  * Duplicate files of the same accident are detected and kept only once
    (the most recently generated report wins); every file is listed in Files_Log.

Usage (Windows example):
    pip install pdfplumber openpyxl pandas pyarrow
    python irad_to_excel.py "D:\\DataSet" "D:\\iRAD_output\\iRAD_Assam.xlsx" --workers 6

Options:
    --workers N     parallel processes (default: CPU count - 1)
    --drop-pii      leave out names, phone numbers, addresses, licence/engine/chassis/
                    policy numbers, blood group, witness statements etc.
    --parquet       also write one .parquet file per sheet next to the workbook
    --csv           also write one .csv file per sheet
    --limit N       only process the first N files (for a quick test)

Parsing is cached in <output>.cache.jsonl, so an interrupted run resumes where it
stopped. Delete the cache file to force a full re-parse.
"""
import argparse, hashlib, json, os, re, sys, time, traceback
from collections import OrderedDict, defaultdict, Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count

try:
    import pdfplumber
except ImportError:
    sys.exit("Missing library: run  pip install pdfplumber openpyxl pandas pyarrow")

PARSER_VERSION = "1.0.0"

# ----------------------------------------------------------------------------------
# Schema: section -> ordered (PDF label -> column name).  Order = column order in Excel.
# ----------------------------------------------------------------------------------
SUMMARY = OrderedDict([
    ("FIR/CSR Number", "fir_csr_number"), ("FIR Date & Time", "fir_datetime_raw"),
    ("Station Name", "station_name"), ("Investigating Officer", "investigating_officer"),
    ("Station Address", "station_address"), ("Field Officer", "field_officer"),
    ("District code", "district_code"), ("District Name", "district_name"),
    ("Act", "act"), ("Section", "legal_section"), ("State Rule", "state_rule"),
])
ACCIDENT = OrderedDict([
    ("Accident Date and Time", "accident_datetime_raw"),
    ("Reporting Date and Time", "reporting_datetime_raw"),
    ("Geolocation", "geolocation_raw"), ("Landmark Name", "landmark_name"),
    ("Location Details", "location_details"), ("Severity", "severity"),
    ("Number of Vehicle(s) involved", "num_vehicles_declared"),
    ("Road Classification", "road_classification"),
    ("Road Name / Street Name", "road_name"), ("Local Body", "local_body"),
    ("Accident Spot", "accident_spot"), ("Collision Type", "collision_type"),
    ("Collision Nature", "collision_nature"), ("Weather Condition", "weather_condition"),
    ("Light Condition", "light_condition"), ("Visibility(Approx.)", "visibility_raw"),
    ("Initial Observation of Accident Scene", "initial_observation"),
    ("Traffic Violation", "traffic_violation"),
    ("Accident Description", "accident_description"),
    ("Property Damage", "property_damage"),
    ("Property Damage Description", "property_damage_description"),
    ("Approximate Damage Value", "approx_damage_value"),
    ("Remedial Measures", "remedial_measures"),
])
VEHICLE = OrderedDict([
    ("Vehicle Registration Number", "vehicle_reg_no"), ("Accused / Victim", "accused_victim"),
    ("Vehicle Category", "vehicle_category"), ("Register Number Status", "reg_number_status"),
    ("Registration Date", "registration_date_raw"), ("Hit & Run", "hit_and_run"),
    ("Disposition", "disposition"), ("Vehicle Damage", "vehicle_damage"),
    ("Owner Name", "owner_name"), ("Owner Father Name", "owner_father_name"),
    ("Owner Address", "owner_address"), ("Vehicle Type", "vehicle_type"),
    ("Load Category", "load_category"), ("Load Condition", "load_condition"),
    ("Colour", "colour"), ("Make & Model", "make_model"), ("Skid Mark", "skid_mark"),
    ("Engine Number", "engine_number"), ("Chassis Number", "chassis_number"),
    ("Vehicle Class", "vehicle_class"), ("Insurance Details", "insurance_company"),
    ("Insurance Policy Number", "insurance_policy_number"),
    ("Insurance Validity", "insurance_validity_raw"),
    ("Vahan Vehicle Category", "vahan_vehicle_category"),
    ("Norms Description", "emission_norms"), ("PUC Certificate Upto", "puc_valid_upto_raw"),
    ("PUC Certificate Number", "puc_certificate_number"), ("Fuel Type", "fuel_type"),
    ("RC Status On", "rc_status_on_raw"), ("Vehicle Laden Weight(GVW)", "gvw_kg"),
    ("Vehicle Un-Laden Weight", "unladen_weight_kg"), ("Number of Cylinder", "num_cylinders"),
    ("Vehicle Cubic Capacity", "cubic_capacity_cc"), ("Seating Capacity", "seating_capacity"),
    ("Standing Capacity", "standing_capacity"), ("Wheelbase", "wheelbase_mm"),
    ("Fitness Validity", "fitness_validity_raw"), ("Year of Manufacture", "manufacture_month_year"),
    ("Tax Validity", "tax_validity_raw"), ("National Permit Number", "national_permit_number"),
    ("National Permit Validity", "national_permit_validity_raw"),
    ("National Permit Issued By", "national_permit_issued_by"), ("Finance Name", "finance_name"),
    ("Owner Serial Number", "owner_serial_number"),
    ("Owner Permanent Address", "owner_permanent_address"),
    ("Owner Present Address", "owner_present_address"),
    ("Vehicle Body Type Description", "body_type"),
])
DRIVER = OrderedDict([
    ("iRAD Victim ID", "victim_id"), ("Is Driver Details Known?", "driver_details_known"),
    ("Licence Number", "licence_number"), ("Driving Licence Type", "licence_type"),
    ("Driving Licence Status", "licence_status"), ("Driver Name", "driver_name"),
    ("Guardian Details", "guardian_details"), ("Nationality", "nationality"),
    ("Injury Type", "injury_type"), ("Education", "education"), ("Occupation", "occupation"),
    ("Cell Phone While Driving?", "cell_phone_while_driving"), ("Severity", "severity"),
    ("Date of Death", "date_of_death_raw"), ("Mode of Hospitalization", "mode_of_hospitalization"),
    ("Seatbelt / Helmet", "seatbelt_helmet"), ("Drunk and Driving", "drunk_driving"),
    ("Hospitalization Delay", "hospitalization_delay"), ("Class of Vehicle", "licensed_vehicle_classes"),
    ("Gender", "gender"), ("Current Mobile number", "current_mobile"),
    ("Mobile Number in Licence", "licence_mobile"), ("Blood Group", "blood_group"),
    ("Age", "age"), ("Marital status", "marital_status"), ("Badge Number", "badge_number"),
    ("Non Transport DL validity", "dl_nontransport_validity_raw"),
    ("Transport DL Validity", "dl_transport_validity_raw"),
    ("Hill DL Validity", "dl_hill_validity_raw"), ("Hazardous DL Validity", "dl_hazardous_validity_raw"),
    ("Permenant address in License", "licence_permanent_address"),
    ("Current Address in License", "licence_current_address"),
    ("Present Address", "present_address"), ("Remarks", "remarks"),
])
PASSENGER = OrderedDict([
    ("Vehicle Registration Number", "vehicle_reg_no"), ("iRAD Victim ID", "victim_id"),
    ("Name", "name"), ("Relation Name", "relation_name"), ("Marital status", "marital_status"),
    ("Gender", "gender"), ("Age", "age"), ("Occupation", "occupation"), ("Severity", "severity"),
    ("Date of Death", "date_of_death_raw"), ("Nationality", "nationality"),
    ("Mode of Hospitalization", "mode_of_hospitalization"),
    ("Hospitalization Delay", "hospitalization_delay"), ("Injury Type", "injury_type"),
    ("Education", "education"), ("Passenger Position", "passenger_position"),
    ("Seatbelt / Helmet", "seatbelt_helmet"), ("Passenger Action", "passenger_action"),
    ("Mobile Number", "mobile_number"), ("Address", "address"),
])
PEDESTRIAN = OrderedDict([
    ("iRAD Victim ID", "victim_id"), ("Vehicle Registration Number", "vehicle_reg_no"),
    ("Name", "name"), ("Relation Name", "relation_name"), ("Marital status", "marital_status"),
    ("Age", "age"), ("Gender", "gender"), ("Severity", "severity"),
    ("Date of Death", "date_of_death_raw"), ("Mode of Hospitalization", "mode_of_hospitalization"),
    ("Hospitalization Delay", "hospitalization_delay"), ("Injury Type", "injury_type"),
    ("Education", "education"), ("Pedestrian Position", "pedestrian_position"),
    ("Pedestrian Action", "pedestrian_action"), ("Mobile Number", "mobile_number"),
    ("Address", "address"), ("Occupation", "occupation"), ("Nationality", "nationality"),
])
TRANSPORT = OrderedDict([
    ("Vehicle Registration Number", "vehicle_reg_no"), ("Registration Number Type", "reg_number_type"),
    ("Registration Date", "registration_date_raw"), ("Owner Serial number", "owner_serial_number"),
    ("Class of Vehicle", "vehicle_class"), ("Type of Vehicle", "vehicle_type"),
    ("Vehicle Use Type", "vehicle_use_type"), ("Make", "make"), ("Model", "model"),
    ("Year of Manufacture", "manufacture_month_year"), ("Age of Vehicle", "vehicle_age_years"),
    ("Engine Number", "engine_number"), ("Chassis Number", "chassis_number"), ("Color", "colour"),
    ("Seat Capacity", "seat_capacity"), ("Un Laden Weight", "unladen_weight_kg"),
    ("Vehicle Max. Speed Limit", "max_speed_limit_kmph"), ("Permit Category", "permit_category"),
    ("Permit Issued By", "permit_issued_by"), ("Permit Number", "permit_number"),
    ("Permit Validity", "permit_validity_raw"), ("Owner Name", "owner_name"),
    ("Owner Address", "owner_address"), ("Vehicle Purpose", "vehicle_purpose"),
    ("RC/FC Validity", "rc_fc_validity_raw"), ("Place of Inspection", "place_of_inspection"),
    ("Insurance Company", "insurance_company"), ("Insurance Policy Number", "insurance_policy_number"),
    ("Insurance Validity", "insurance_validity_raw"), ("Inspection Date", "inspection_datetime_raw"),
    ("Driving license Submited?", "driving_licence_submitted"),
    ("Length of Vehicle In MM (After Crash)", "length_after_crash_mm"),
    ("Width of Vehicle In MM (After Crash)", "width_after_crash_mm"),
    ("Height of Vehicle In MM (After Crash)", "height_after_crash_mm"),
    ("Brake Type", "brake_type"), ("Condition of Brake", "brake_condition"),
    ("Condition of Foot Brake", "foot_brake_condition"), ("Condition of Hand Brake", "hand_brake_condition"),
    ("Brakes Even or Not", "brakes_even"), ("Mechanical Failure Status", "mechanical_failure_status"),
    ("Mechanical", "mechanical_failure_detail"), ("Tyre Condition", "tyre_condition"),
    ("Accident Due To?", "accident_due_to"), ("Tax Details", "tax_details"),
    ("Steering Type", "steering_type"), ("Handle/Steering Condition", "steering_condition"),
    ("Damage Status", "damage_status"), ("Description of Vehicle Damage", "damage_description"),
    ("Skid Mark", "skid_mark"), ("Check Report Issued?", "check_report_issued"),
    ("Check Report Number", "check_report_number"), ("Report issued Date", "check_report_date_raw"),
    ("CFX Issued?", "cfx_issued"), ("CFX Number", "cfx_number"), ("CFX Date", "cfx_date_raw"),
    ("FC Status", "fc_status"), ("FC Validity", "fc_validity_raw"), ("RLW", "rlw_kg"),
    ("Pollution Under Control Certificate Validity", "puc_validity_raw"),
    ("Vehicle Defect Type", "vehicle_defect_type"), ("Manoeuvre", "manoeuvre"),
    ("Condition of Wheels", "wheel_condition"), ("Condition of Wipers", "wiper_condition"),
    ("Condition of Mirrors", "mirror_condition"), ("Whether Vehicle Modified", "vehicle_modified"),
    ("Rear Parking Sensor", "rear_parking_sensor"),
    ("Rear Parking Sensor :: functional", "rear_parking_sensor_functional"),
    ("Whether scratch marks found ?", "scratch_marks_found"),
    ("Estimated cost of damage (including internal & external damage)", "estimated_damage_cost"),
    ("Cause of Accident", "cause_of_accident"), ("National Permit Number", "national_permit_number"),
    ("National Permit Validity", "national_permit_validity_raw"),
    ("National Permit Issued By", "national_permit_issued_by"), ("Finance Name", "finance_name"),
    ("Owner Permanent Address", "owner_permanent_address"),
    ("Owner Present Address", "owner_present_address"),
    ("Vehicle Body Type Description", "body_type"), ("Paint transfer found", "paint_transfer_found"),
    ("Location of paint transfer", "paint_transfer_location"),
    ("Color of paint transfer", "paint_transfer_colour"),
    ("Installed CNG / LPG kit", "cng_lpg_kit_installed"),
    ("Horn installed and functional?", "horn_functional"),
    ("Brake lights & other lights functional?", "lights_functional"),
    ("Vehicle had faulty number plate?", "faulty_number_plate"),
    ("Whether the vehicle fitted with airbags?", "airbags_fitted"),
    ("Airbags Deployed?", "airbags_deployed"), ("If not deployed, state reasons", "airbags_not_deployed_reason"),
    ("Vehicle had tinted glasses?", "tinted_glasses"),
    ("For educational institutions bus, whether the vehicle was fitted with the doors that can be shut & whether the vehicle had a suitable inscription to indicate that they are in the duty of an educational institute?", "school_bus_doors_and_inscription"),
    ("Speed Limiter devices in case of PSV (Commercial vehicle)", "speed_limiter_device"),
    ("Speed Limiter devices in case of PSV (Commercial vehicle) :: functional", "speed_limiter_functional"),
    ("Front Parking Sensors", "front_parking_sensors"),
    ("Front Parking Sensors :: functional", "front_parking_sensors_functional"),
    ("Vehicle Location Tracking (VLT) Devices", "vlt_device"),
    ("Vehicle Location Tracking (VLT) Devices :: functional", "vlt_functional"),
    ("Run protection device and Side under run protection device", "underrun_protection_device"),
    ("Bull Bars", "bull_bars"), ("Reflective Tapes", "reflective_tapes"),
    ("WindScreen Safety", "windscreen_safety"),
])
ROAD = OrderedDict([
    ("Area Type", "area_type"), ("Road Classification", "road_classification"),
    ("Road Owning Agency", "road_owning_agency"), ("Road Number", "road_number"),
    ("Road Name / Street Name", "road_name"), ("Type of Structure", "structure_type"),
    ("Type of Road Surface", "road_surface_type"), ("Surface Condition", "surface_condition"),
    ("Type of Carriageway", "carriageway_type"), ("Road Width (in metre)", "road_width_m"),
    ("Accident Location", "accident_location_geometry"), ("Road Chainage (in metre)", "road_chainage_m"),
    ("Sight Distance", "sight_distance"), ("Speed Limit (in kmph)", "speed_limit_kmph"),
    ("Road Margins", "road_margins"), ("Road Shoulder Type", "shoulder_type"),
    ("Type of Terrain", "terrain_type"), ("Type of Surface Gradient", "gradient_type"),
    ("Physical Divider / Barrier Type", "divider_barrier_type"), ("Type of Median", "median_type"),
    ("Pedestrian Infrastructure", "pedestrian_infrastructure"),
    ("Quality of Pedestrian Infrastructure", "pedestrian_infrastructure_quality"),
    ("Ongoing Road Work", "ongoing_road_work"), ("Road Markings", "road_markings"),
    ("Road Sign Board", "road_sign_board"),
    ("Contributing Factors of Road Accident", "contributing_factors"),
    ("Short-Term Remedial Measures", "short_term_remedial_measures"),
    ("Long-Term Remedial Measures", "long_term_remedial_measures"),
])
HOSPITAL = OrderedDict([
    ("Hospital Name", "hospital_name"), ("Mode of Hospitalization", "mode_of_hospitalization"),
    ("Patient Type", "patient_type"), ("Name of the Patient", "patient_name"),
    ("Patient Age", "patient_age"), ("Gender", "gender"), ("Guardian Details", "guardian_details"),
    ("Doctor Name", "doctor_name"), ("Severity Type", "severity_type"),
    ("Type of Injury", "injury_type"), ("Nature of Injury", "injury_nature"),
    ("Identification Mark1", "identification_mark_1"), ("Identification Mark2", "identification_mark_2"),
    ("BP Details", "bp"), ("Pulse Rate", "pulse_rate"), ("Respiratory Rate", "respiratory_rate"),
    ("Oxygen Level", "oxygen_level"), ("Temperature", "temperature"), ("Breathing", "breathing"),
    ("Xrays done", "xrays_done"), ("CT Scan done", "ct_scan_done"),
    ("Informant Name", "informant_name"), ("Informant Address", "informant_address"),
    ("Informant Contact No", "informant_contact"), ("Type of Person", "person_type"),
])
TMS = OrderedDict([
    ("TMS Patient ID", "tms_patient_id"), ("Victim ID", "victim_id"), ("Hospital Name", "hospital_name"),
    ("State Name", "state_name"), ("District Name", "district_name"), ("Station Name", "station_name"),
    ("Updated By", "updated_by"), ("Requested Date Time", "requested_datetime_raw"),
    ("Approved Date Time", "approved_datetime_raw"), ("Approve Flag", "approve_flag"),
])
WITNESS_COLS = OrderedDict([
    ("Name", "name"), ("Gender", "gender"), ("Age", "age"), ("Occupation", "occupation"),
    ("Mobile Number", "mobile_number"), ("Address", "address"), ("Statement", "statement"),
])

# section heading in PDF -> (sheet, schema, label that always starts a new record)
SECTIONS = OrderedDict([
    ("Accident Details",  ("Accidents", ACCIDENT, None)),
    ("Vehicle Details",   ("Vehicles", VEHICLE, "Vehicle Registration Number")),
    ("Driver Details",    ("Drivers", DRIVER, "iRAD Victim ID")),
    ("Passenger",         ("Passengers", PASSENGER, "iRAD Victim ID")),
    ("Pedestrian",        ("Pedestrians", PEDESTRIAN, "iRAD Victim ID")),
    ("Witness",           ("Witnesses", WITNESS_COLS, None)),
    ("Transport Details", ("Transport_Inspections", TRANSPORT, "Vehicle Registration Number")),
    ("Road Details",      ("Road_Details", ROAD, "Area Type")),
    ("Hospital Details",  ("Hospital_Details", HOSPITAL, "Hospital Name")),
    ("TMS Details",       ("TMS_Details", TMS, "TMS Patient ID")),
])
ENTITY_SECTIONS = {"Vehicle Details", "Driver Details", "Passenger", "Pedestrian",
                   "Transport Details", "Road Details", "Hospital Details", "TMS Details"}

PII_COLUMNS = {
    "Accidents": {"investigating_officer", "field_officer"},
    "Vehicles": {"owner_name", "owner_father_name", "owner_address", "engine_number", "chassis_number",
                 "insurance_policy_number", "puc_certificate_number", "owner_permanent_address",
                 "owner_present_address", "entity_header"},
    "Drivers": {"licence_number", "driver_name", "guardian_details", "current_mobile", "licence_mobile",
                "blood_group", "badge_number", "licence_permanent_address", "licence_current_address",
                "present_address", "entity_header"},
    "Passengers": {"name", "relation_name", "mobile_number", "address", "entity_header"},
    "Pedestrians": {"name", "relation_name", "mobile_number", "address", "entity_header"},
    "Witnesses": {"name", "mobile_number", "address", "statement"},
    "Transport_Inspections": {"engine_number", "chassis_number", "owner_name", "owner_address",
                              "insurance_policy_number", "permit_number", "owner_permanent_address",
                              "owner_present_address", "entity_header"},
    "Hospital_Details": {"patient_name", "guardian_details", "doctor_name", "identification_mark_1",
                         "identification_mark_2", "informant_name", "informant_address",
                         "informant_contact", "entity_header"},
    "TMS_Details": {"updated_by"},
}

NULL_TOKENS = {"", "NA", "N/A", "N.A.", "NULL", "NONE", "-", "--"}
ASSAM_BBOX = (24.0, 28.3, 89.6, 96.2)   # lat_min, lat_max, lon_min, lon_max (generous)

def norm_label(s):
    return re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()

def norm_value(s):
    if s is None:
        return None
    v = re.sub(r"[ \t]+", " ", s.replace("\r", "").replace("\n", " ")).strip()
    return None if v.upper() in NULL_TOKENS else v

LABEL_LOOKUP = {}
for _sec, (_sheet, _schema, _) in SECTIONS.items():
    LABEL_LOOKUP[_sec] = {norm_label(k): v for k, v in _schema.items()}
SUMMARY_LOOKUP = {norm_label(k): v for k, v in SUMMARY.items()}

# ----------------------------------------------------------------------------------
# PDF -> ordered event stream -> records
# ----------------------------------------------------------------------------------
def page_events(page):
    tables = page.find_tables()
    boxes = [t.bbox for t in tables]
    ev = [(t.bbox[1], 0, "T", t) for t in tables]
    for ln in page.extract_text_lines():
        inside = any(b[0] - 1 <= ln["x0"] and ln["x1"] <= b[2] + 1 and
                     b[1] - 1 <= ln["top"] and ln["bottom"] <= b[3] + 1 for b in boxes)
        if not inside:
            ev.append((ln["top"], 1, "L", ln["text"].strip()))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev

def parse_pdf(path):
    """Returns a dict with raw per-section records for one file (no type conversion)."""
    out = {"source_file": path, "accident_id": None, "generated_at_raw": None, "pages": 0,
           "sections_present": [], "summary": {}, "accident": {}, "casualties": {},
           "entities": defaultdict(list), "witnesses": [], "unknown_labels": [], "notes": []}
    section = "Summary"
    current = None          # current entity dict
    pending_header = None
    last_label = None
    with pdfplumber.open(path) as pdf:
        out["pages"] = len(pdf.pages)
        for page in pdf.pages:
            for _, _, kind, obj in page_events(page):
                if kind == "L":
                    txt = obj
                    m = re.match(r"Accident Summary\s*-\s*(\S+)", txt)
                    if m and not out["accident_id"]:
                        out["accident_id"] = m.group(1)
                        continue
                    g = re.search(r"Date of Generation\s+(.+?)\s+Page\s+\d+", txt)
                    if g:
                        out["generated_at_raw"] = out["generated_at_raw"] or g.group(1).strip()
                        continue
                    if txt in SECTIONS:
                        section = txt
                        current = None
                        pending_header = None
                        if txt not in out["sections_present"]:
                            out["sections_present"].append(txt)
                        continue
                    if section in ENTITY_SECTIONS and "|" in txt:
                        pending_header = txt      # "REG | OWNER", "LICENCE | REG", "1) Name | ID"
                        current = None
                    continue

                # ---- table ----
                for row in obj.extract():
                    cells = list(row)
                    ncell = len(cells)
                    if section == "Summary" and ncell >= 2:
                        for i in range(0, ncell - 1, 2):
                            lab = norm_label(cells[i])
                            if not lab:
                                continue
                            col = SUMMARY_LOOKUP.get(lab)
                            if col:
                                out["summary"][col] = norm_value(cells[i + 1])
                            else:
                                out["unknown_labels"].append(("Summary", lab, norm_value(cells[i + 1])))
                        continue
                    if section == "Accident Details" and ncell == 6:
                        lab = norm_label(cells[0])
                        if lab.startswith("Number of Persons involved"):
                            continue
                        if lab.startswith("Number of Animals involved"):
                            out["casualties"]["animals_total"] = norm_value(cells[5])
                            continue
                        if lab in ("Driver", "Passenger", "Pedestrian", "Total"):
                            key = lab.lower()
                            for j, cat in enumerate(["killed", "grievous", "minor", "no_injury", "total"], start=1):
                                out["casualties"][f"{cat}_{key}"] = norm_value(cells[j])
                        continue
                    if section == "Witness" and ncell >= 7:
                        lab0 = norm_label(cells[0])
                        if lab0 == "Name" and norm_label(cells[1]) == "Gender":
                            continue      # header row
                        w = OrderedDict()
                        for (lab, col), c in zip(WITNESS_COLS.items(), cells[:7]):
                            w[col] = norm_value(c)
                        out["witnesses"].append(w)
                        continue
                    if ncell < 2:
                        continue
                    lab = norm_label(cells[0])
                    val = norm_value(cells[1])
                    if ncell > 2 and any(norm_value(c) for c in cells[2:]):
                        out["notes"].append(f"{section}: row with {ncell} cells, extra cells ignored: {cells}")
                    if not lab:
                        # continuation of previous value split across a page break
                        if current is not None and last_label and val:
                            current[last_label] = (current.get(last_label) or "") + " " + val
                            out["notes"].append(f"{section}: continuation row appended to '{last_label}'")
                        continue
                    if section not in LABEL_LOOKUP:
                        out["unknown_labels"].append((section, lab, val))
                        continue
                    lookup = LABEL_LOOKUP[section]
                    # disambiguate repeated 'Whether functional or not?' questions
                    if re.match(r"whether function(al)? or not\??$", lab, re.I) and last_label:
                        lab = f"{last_label} :: functional"
                    col = lookup.get(lab)
                    if section == "Accident Details":
                        if col:
                            out["accident"][col] = val
                        else:
                            out["unknown_labels"].append((section, lab, val))
                        last_label = lab
                        continue
                    sheet, schema, start_label = SECTIONS[section]
                    new_needed = (current is None or
                                  (start_label and lab == norm_label(start_label) and start_label_seen(current, schema, start_label)) or
                                  (col and col in current))
                    if new_needed:
                        current = OrderedDict()
                        current["entity_header"] = pending_header
                        pending_header = None
                        out["entities"][section].append(current)
                    if col:
                        current[col] = val
                    else:
                        current.setdefault("_unknown", []).append((lab, val))
                        out["unknown_labels"].append((section, lab, val))
                    last_label = lab if not lab.endswith(":: functional") else last_label
    out["entities"] = dict(out["entities"])
    return out

def start_label_seen(current, schema, start_label):
    return schema[start_label] in current

def file_worker(path):
    t0 = time.time()
    try:
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        rec = parse_pdf(path)
        rec["sha256"] = sha
        rec["status"] = "parsed" if rec["accident_id"] else "error: no accident id found"
    except Exception as e:
        rec = {"source_file": path, "status": f"error: {type(e).__name__}: {e}",
               "traceback": traceback.format_exc(limit=3)}
    rec["parse_seconds"] = round(time.time() - t0, 3)
    rec["parser_version"] = PARSER_VERSION
    return rec

# ----------------------------------------------------------------------------------
# Typed helpers (explicit formats only)
# ----------------------------------------------------------------------------------
DT_FORMATS = ["%d-%b-%Y : %I:%M %p", "%d-%b-%Y - %I:%M %p", "%d %b %Y %I:%M %p",
              "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %I:%M %p"]
D_FORMATS = ["%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y"]

def parse_dt(s):
    if not s:
        return None
    s2 = re.sub(r"\s+", " ", s.strip())
    for f in DT_FORMATS:
        try:
            return datetime.strptime(s2, f)
        except ValueError:
            pass
    return None

def parse_date(s):
    if not s:
        return None
    s2 = s.strip()
    for f in D_FORMATS:
        try:
            return datetime.strptime(s2, f).date()
        except ValueError:
            pass
    m = re.match(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}", s2)   # ISO timestamp (seen in Transport section)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    dt = parse_dt(s2)
    return dt.date() if dt else None

def parse_num(s):
    if s is None:
        return None
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:meters?|m|kmph|km/h|kg|mm)?\s*", s, re.I)
    return float(m.group(1)) if m else None

def parse_int(s):
    n = parse_num(s)
    return int(n) if n is not None and float(n).is_integer() else None

def parse_latlon(s):
    if not s:
        return None, None
    m = re.search(r"Lat\s*:\s*(-?\d+(?:\.\d+)?)\s*Lon\s*:\s*(-?\d+(?:\.\d+)?)", s)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

def parse_month_year(s):
    if not s:
        return None, None
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{4})\s*", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    return None, None

# which raw columns get typed companions
DATE_COLS = {  # sheet -> {raw_col: (typed_col, kind)}
    "Accidents": {"accident_datetime_raw": ("accident_datetime", "dt"),
                  "reporting_datetime_raw": ("reporting_datetime", "dt"),
                  "fir_datetime_raw": ("fir_datetime", "dt")},
    "Vehicles": {"registration_date_raw": ("registration_date", "d"),
                 "insurance_validity_raw": ("insurance_validity", "d"),
                 "puc_valid_upto_raw": ("puc_valid_upto", "d"),
                 "rc_status_on_raw": ("rc_status_on", "d"),
                 "fitness_validity_raw": ("fitness_validity", "d"),
                 "tax_validity_raw": ("tax_validity", "d"),
                 "national_permit_validity_raw": ("national_permit_validity", "d")},
    "Drivers": {"date_of_death_raw": ("date_of_death", "dt"),
                "dl_nontransport_validity_raw": ("dl_nontransport_validity", "d"),
                "dl_transport_validity_raw": ("dl_transport_validity", "d"),
                "dl_hill_validity_raw": ("dl_hill_validity", "d"),
                "dl_hazardous_validity_raw": ("dl_hazardous_validity", "d")},
    "Passengers": {"date_of_death_raw": ("date_of_death", "dt")},
    "Pedestrians": {"date_of_death_raw": ("date_of_death", "dt")},
    "Transport_Inspections": {"registration_date_raw": ("registration_date", "d"),
                              "permit_validity_raw": ("permit_validity", "d"),
                              "rc_fc_validity_raw": ("rc_fc_validity", "d"),
                              "insurance_validity_raw": ("insurance_validity", "d"),
                              "inspection_datetime_raw": ("inspection_datetime", "dt"),
                              "check_report_date_raw": ("check_report_date", "d"),
                              "cfx_date_raw": ("cfx_date", "d"),
                              "fc_validity_raw": ("fc_validity", "d"),
                              "puc_validity_raw": ("puc_validity", "d"),
                              "national_permit_validity_raw": ("national_permit_validity", "d")},
    "TMS_Details": {"requested_datetime_raw": ("requested_datetime", "dt"),
                    "approved_datetime_raw": ("approved_datetime", "dt")},
}
NUM_COLS = {
    "Accidents": {"num_vehicles_declared": "int"},
    "Vehicles": {"gvw_kg": "num", "unladen_weight_kg": "num", "num_cylinders": "int",
                 "cubic_capacity_cc": "num", "seating_capacity": "int", "standing_capacity": "int",
                 "wheelbase_mm": "num", "owner_serial_number": "int"},
    "Drivers": {"age": "int"}, "Passengers": {"age": "int"}, "Pedestrians": {"age": "int"},
    "Witnesses": {"age": "int"},
    "Transport_Inspections": {"vehicle_age_years": "int", "seat_capacity": "int",
                              "unladen_weight_kg": "num", "max_speed_limit_kmph": "num",
                              "rlw_kg": "num", "owner_serial_number": "int",
                              "length_after_crash_mm": "num", "width_after_crash_mm": "num",
                              "height_after_crash_mm": "num"},
    "Road_Details": {"road_width_m": "num", "road_chainage_m": "num", "speed_limit_kmph": "num"},
    "Hospital_Details": {"patient_age": "int"},
}
NON_NUMERIC_OK = {"Not Known", "Not Applicable", "Unknown"}

# ----------------------------------------------------------------------------------
# Build tables + validation
# ----------------------------------------------------------------------------------
def build(records, drop_pii=False):
    issues = []
    def issue(level, acc, sheet, field, value, msg, src):
        issues.append(OrderedDict(level=level, accident_id=acc, sheet=sheet, field=field,
                                  value=None if value is None else str(value)[:500],
                                  issue=msg, source_file=src))

    files_log = []
    # --- duplicate resolution: one record per accident_id, latest generation wins
    by_id = defaultdict(list)
    for r in records:
        if r.get("status") == "parsed":
            by_id[r["accident_id"]].append(r)
    chosen = {}
    for aid, rs in by_id.items():
        def gen_key(r):
            return parse_dt(r.get("generated_at_raw") or "") or datetime.min
        rs_sorted = sorted(rs, key=lambda r: (gen_key(r), r["source_file"]), reverse=True)
        chosen[aid] = rs_sorted[0]
        content_hashes = {json.dumps([r["summary"], r["accident"], r["casualties"], r["entities"],
                                      r["witnesses"]], sort_keys=True, default=str) for r in rs}
        for r in rs_sorted[1:]:
            r["_dup_status"] = ("duplicate (identical content)" if len(content_hashes) == 1
                                else "duplicate (superseded by newer report)")
            issue("info", aid, "Files_Log", "source_file", r["source_file"], r["_dup_status"] +
                  f"; kept {chosen[aid]['source_file']}", r["source_file"])

    for r in records:
        st = r.get("status", "")
        if st == "parsed":
            st = "kept" if chosen.get(r["accident_id"]) is r else r.get("_dup_status", "duplicate")
        p = r["source_file"]
        parts = re.split(r"[\\/]", p)
        files_log.append(OrderedDict(
            source_file=p, status=st, accident_id=r.get("accident_id"),
            folder_year=parts[-3] if len(parts) >= 3 else None,
            folder_month=parts[-2] if len(parts) >= 2 else None,
            pages=r.get("pages"), generated_at=parse_dt(r.get("generated_at_raw") or ""),
            sections_present="; ".join(r.get("sections_present", [])),
            unknown_label_count=len(r.get("unknown_labels", [])),
            parser_notes=" | ".join(r.get("notes", []))[:2000] or None,
            sha256=r.get("sha256"), parse_seconds=r.get("parse_seconds"),
            parser_version=r.get("parser_version")))
        if st.startswith("error"):
            issue("error", r.get("accident_id"), "Files_Log", "status", None, st, p)

    sheets = OrderedDict((n, []) for n in [
        "Accidents", "Vehicles", "Drivers", "Passengers", "Pedestrians", "Witnesses",
        "Transport_Inspections", "Road_Details", "Hospital_Details", "TMS_Details"])

    for aid in sorted(chosen):
        r = chosen[aid]
        src = r["source_file"]
        parts = re.split(r"[\\/]", src)
        fname = parts[-1]
        m = re.search(r"accid(\d+)", fname)
        if m and m.group(1) != aid:
            issue("warning", aid, "Accidents", "accident_id", fname,
                  "accident id in file name differs from id printed in report", src)

        # ---------- Accidents row
        a = OrderedDict()
        a["accident_id"] = aid
        a["source_file"] = src
        a["folder_year"] = parts[-3] if len(parts) >= 3 else None
        a["folder_month"] = parts[-2] if len(parts) >= 2 else None
        a["report_generated_at"] = parse_dt(r.get("generated_at_raw") or "")
        a["pages"] = r.get("pages")
        a["sections_present"] = "; ".join(r.get("sections_present", []))
        for col in SUMMARY.values():
            a[col] = r["summary"].get(col)
        for col in ACCIDENT.values():
            a[col] = r["accident"].get(col)
        # typed helpers
        acc_dt = parse_dt(a["accident_datetime_raw"])
        rep_dt = parse_dt(a["reporting_datetime_raw"])
        a["accident_datetime"] = acc_dt
        a["accident_year"] = acc_dt.year if acc_dt else None
        a["accident_month"] = acc_dt.month if acc_dt else None
        a["accident_hour"] = acc_dt.hour if acc_dt else None
        a["accident_weekday"] = acc_dt.strftime("%A") if acc_dt else None
        a["reporting_datetime"] = rep_dt
        a["reporting_delay_hours"] = (round((rep_dt - acc_dt).total_seconds() / 3600, 2)
                                      if acc_dt and rep_dt else None)
        a["fir_datetime"] = parse_dt(a["fir_datetime_raw"])
        lat, lon = parse_latlon(a["geolocation_raw"])
        a["latitude"], a["longitude"] = lat, lon
        a["visibility_m"] = parse_num(a["visibility_raw"])
        a["num_vehicles_declared_int"] = parse_int(a["num_vehicles_declared"])
        # casualties
        cas = r["casualties"]
        for key in ["driver", "passenger", "pedestrian", "total"]:
            for cat in ["killed", "grievous", "minor", "no_injury", "total"]:
                v = cas.get(f"{cat}_{key}")
                iv = parse_int(v) if v is not None else 0      # blank cell in the table = 0
                if v is not None and iv is None:
                    issue("error", aid, "Accidents", f"{cat}_{key}", v, "non-numeric casualty count", src)
                a[f"{cat}_{key}"] = iv
        av = cas.get("animals_total")
        a["animals_total"] = parse_int(av) if av is not None else 0
        # entity counts
        ents = r["entities"]
        a["n_vehicle_records"] = len(ents.get("Vehicle Details", []))
        a["n_driver_records"] = len(ents.get("Driver Details", []))
        a["n_passenger_records"] = len(ents.get("Passenger", []))
        a["n_pedestrian_records"] = len(ents.get("Pedestrian", []))
        a["n_witness_records"] = len(r["witnesses"])
        a["n_transport_inspections"] = len(ents.get("Transport Details", []))
        a["has_road_details"] = bool(ents.get("Road Details"))
        a["has_hospital_details"] = bool(ents.get("Hospital Details"))
        sheets["Accidents"].append(a)

        # ---------- accident-level validation
        if not a["accident_datetime_raw"]:
            issue("warning", aid, "Accidents", "accident_datetime_raw", None, "missing accident date/time", src)
        elif not acc_dt:
            issue("error", aid, "Accidents", "accident_datetime_raw", a["accident_datetime_raw"],
                  "accident date/time in unexpected format (typed column left blank)", src)
        if a["reporting_datetime_raw"] and not rep_dt:
            issue("error", aid, "Accidents", "reporting_datetime_raw", a["reporting_datetime_raw"],
                  "reporting date/time in unexpected format", src)
        if acc_dt and rep_dt and rep_dt < acc_dt:
            issue("warning", aid, "Accidents", "reporting_datetime", a["reporting_datetime_raw"],
                  "reported BEFORE the accident time", src)
        gen = a["report_generated_at"]
        if acc_dt and gen and acc_dt > gen:
            issue("warning", aid, "Accidents", "accident_datetime", a["accident_datetime_raw"],
                  "accident time is after report generation time", src)
        if a["geolocation_raw"] and lat is None:
            issue("error", aid, "Accidents", "geolocation_raw", a["geolocation_raw"], "could not parse lat/lon", src)
        if lat is not None:
            if lat == 0 and lon == 0:
                issue("warning", aid, "Accidents", "geolocation_raw", a["geolocation_raw"], "coordinates are 0,0", src)
            elif not (ASSAM_BBOX[0] <= lat <= ASSAM_BBOX[1] and ASSAM_BBOX[2] <= lon <= ASSAM_BBOX[3]):
                issue("warning", aid, "Accidents", "geolocation_raw", a["geolocation_raw"],
                      "coordinates outside Assam bounding box", src)
        if a["visibility_raw"] and a["visibility_m"] is None and a["visibility_raw"] not in NON_NUMERIC_OK:
            issue("info", aid, "Accidents", "visibility_raw", a["visibility_raw"], "visibility not numeric", src)
        # casualty arithmetic
        for key in ["driver", "passenger", "pedestrian", "total"]:
            s = sum(a[f"{c}_{key}"] or 0 for c in ["killed", "grievous", "minor", "no_injury"])
            if s != (a[f"total_{key}"] or 0):
                issue("warning", aid, "Accidents", f"total_{key}", a[f"total_{key}"],
                      f"row total {a[f'total_{key}']} != sum of categories {s}", src)
        for c in ["killed", "grievous", "minor", "no_injury", "total"]:
            s = sum(a[f"{c}_{k}"] or 0 for k in ["driver", "passenger", "pedestrian"])
            if s != (a[f"{c}_total"] or 0):
                issue("warning", aid, "Accidents", f"{c}_total", a[f"{c}_total"],
                      f"column total {a[f'{c}_total']} != driver+passenger+pedestrian {s}", src)
        sev = (a["severity"] or "").lower()
        killed = a["killed_total"] or 0
        if sev == "fatal" and killed == 0:
            issue("warning", aid, "Accidents", "severity", a["severity"], "severity Fatal but 0 killed in casualty table", src)
        if killed > 0 and sev and sev != "fatal":
            issue("warning", aid, "Accidents", "severity", a["severity"], f"{killed} killed but severity is not Fatal", src)
        nv = a["num_vehicles_declared_int"]
        if nv is not None and a["n_vehicle_records"] and nv != a["n_vehicle_records"]:
            issue("info", aid, "Accidents", "num_vehicles_declared", nv,
                  f"declared vehicles {nv} != vehicle records in report {a['n_vehicle_records']}", src)
        if (a["light_condition"] or "").lower().startswith("night") and "sunny" in (a["weather_condition"] or "").lower():
            issue("info", aid, "Accidents", "weather_condition", a["weather_condition"],
                  "weather 'Sunny' recorded with light condition Night", src)
        for (sec, lab, val) in r.get("unknown_labels", []):
            issue("warning", aid, sec, lab, val, "label not in schema (value kept only here)", src)

        # ---------- entity sheets
        for sec, (sheet, schema, _) in SECTIONS.items():
            if sec in ("Accident Details", "Witness"):
                continue
            prev = None
            for i, e in enumerate(ents.get(sec, []), start=1):
                row = OrderedDict(accident_id=aid, record_seq=i)
                hdr = e.get("entity_header")
                row["entity_header"] = hdr
                for col in schema.values():
                    row[col] = e.get(col)
                if hdr and "|" in hdr:
                    left, right = [x.strip() for x in hdr.split("|", 1)]
                    if sec == "Driver Details":
                        row["vehicle_reg_no_from_header"] = right or None
                    elif sec in ("Passenger", "Pedestrian"):
                        row["header_ref"] = right or None
                for lab, val in e.get("_unknown", []):
                    pass  # already logged
                # typed helpers
                for raw, (typed, kind) in DATE_COLS.get(sheet, {}).items():
                    v = row.get(raw)
                    t = (parse_dt(v) if kind == "dt" else parse_date(v)) if v else None
                    if kind == "dt" and v and t is None:
                        t2 = parse_date(v)
                        t = datetime.combine(t2, datetime.min.time()) if t2 else None
                    row[typed] = t
                    if v and t is None and v not in NON_NUMERIC_OK:
                        issue("info", aid, sheet, raw, v, "date not in a recognised format (typed column blank)", src)
                for c, kind in NUM_COLS.get(sheet, {}).items():
                    v = row.get(c)
                    if v is None:
                        continue
                    n = parse_int(v) if kind == "int" else parse_num(v)
                    row[c + "_num"] = n
                    if n is None and v not in NON_NUMERIC_OK:
                        issue("info", aid, sheet, c, v, "value not numeric", src)
                if sheet == "Vehicles":
                    ym = parse_month_year(row.get("manufacture_month_year"))
                    row["manufacture_year"], row["manufacture_month"] = ym
                    # iRAD print bug: make & model of the previous vehicle prefixed to this one
                    mm = row.get("make_model")
                    row["make_model_clean"] = mm
                    if prev and mm and prev.get("make_model") and mm != prev["make_model"] \
                            and mm.startswith(prev["make_model"]):
                        row["make_model_clean"] = mm[len(prev["make_model"]):].strip() or mm
                        issue("info", aid, sheet, "make_model", mm,
                              "value starts with previous vehicle's make & model (source print artefact); "
                              "make_model_clean has the prefix removed", src)
                if sheet in ("Drivers", "Passengers", "Pedestrians"):
                    ag = row.get("age_num")
                    if ag is not None and not (0 <= ag <= 110):
                        issue("warning", aid, sheet, "age", row.get("age"), "age outside 0-110", src)
                    dod = row.get("date_of_death")
                    if dod and acc_dt and dod < acc_dt.replace(second=0) and dod.date() < acc_dt.date():
                        issue("warning", aid, sheet, "date_of_death", row.get("date_of_death_raw"),
                              "date of death before accident date", src)
                    if dod and acc_dt:
                        row["hours_accident_to_death"] = round((dod - acc_dt).total_seconds() / 3600, 2)
                    sv = (row.get("severity") or "").lower()
                    if sv == "fatal" and not row.get("date_of_death_raw"):
                        pass
                sheets[sheet].append(row)
                prev = row
        for i, w in enumerate(r["witnesses"], start=1):
            row = OrderedDict(accident_id=aid, record_seq=i)
            row.update(w)
            row["age_num"] = parse_int(w.get("age")) if w.get("age") else None
            sheets["Witnesses"].append(row)

    if drop_pii:
        for sheet, cols in PII_COLUMNS.items():
            for row in sheets.get(sheet, []):
                for c in cols:
                    row.pop(c, None)
        for row in issues:
            if row["field"] and any(row["field"] in cols for cols in PII_COLUMNS.values()):
                row["value"] = "[removed: --drop-pii]"

    sheets["Validation_Issues"] = issues
    sheets["Files_Log"] = files_log
    return sheets

# ----------------------------------------------------------------------------------
# Data dictionary + writing
# ----------------------------------------------------------------------------------
def data_dictionary(sheets):
    src_label = {}
    for sec, (sheet, schema, _) in SECTIONS.items():
        for lab, col in schema.items():
            src_label[(sheet, col)] = f"[{sec}] {lab}"
    for lab, col in SUMMARY.items():
        src_label[("Accidents", col)] = f"[Header] {lab}"
    derived_desc = {
        "accident_id": "Accident ID printed as 'Accident Summary - <id>'",
        "source_file": "Path of the PDF this row came from",
        "record_seq": "Order of this record within its accident report",
        "entity_header": "Heading line above the record in the PDF (e.g. 'REG | OWNER')",
        "report_generated_at": "'Date of Generation' footer of the PDF",
        "reporting_delay_hours": "reporting_datetime - accident_datetime, in hours",
        "latitude": "Parsed from geolocation_raw", "longitude": "Parsed from geolocation_raw",
        "visibility_m": "Number parsed from visibility_raw ('50 meters' -> 50)",
        "make_model_clean": "make_model with an exact copy of the previous vehicle's make & model removed from the start (source print artefact); otherwise identical",
        "hours_accident_to_death": "date_of_death - accident_datetime, in hours",
        "vehicle_reg_no_from_header": "Vehicle registration after '|' in the driver heading",
        "folder_year": "Year folder the file was found in", "folder_month": "Month folder the file was found in",
    }
    rows = []
    for sheet, data in sheets.items():
        cols = []
        for r in data:
            for c in r:
                if c not in cols:
                    cols.append(c)
        for c in cols:
            if (sheet, c) in src_label:
                origin, desc = "observed (verbatim from PDF)", src_label[(sheet, c)]
            elif c.startswith(("killed_", "grievous_", "minor_", "no_injury_", "total_")) or c == "animals_total":
                origin, desc = "observed (casualty table)", "[Accident Details] Number of persons table: category_roleof person; blank cell = 0"
            elif c.endswith("_num") or c in ("accident_datetime", "reporting_datetime", "fir_datetime") or \
                    any(c == t for m in DATE_COLS.values() for (t, _) in m.values()):
                origin, desc = "derived (typed copy of raw column)", "Parsed with explicit formats; blank if the raw value did not match"
            else:
                origin, desc = "derived", derived_desc.get(c, "")
            rows.append(OrderedDict(sheet=sheet, column=c, origin=origin, source_or_description=desc))
    return rows

def write_outputs(sheets, out_xlsx, args, n_files):
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    ver = sheets.pop("_verify", None)
    dd = data_dictionary(sheets)
    lv = Counter(i["level"] for i in sheets["Validation_Issues"])
    fl = Counter(f["status"].split(":")[0] for f in sheets["Files_Log"])
    summary = [OrderedDict(item=k, value=v) for k, v in [
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("parser_version", PARSER_VERSION), ("input_folder", args.input),
        ("pdf_files_found", n_files), *[(f"files_{k}", v) for k, v in sorted(fl.items())],
        *[(f"rows_{k}", len(v)) for k, v in sheets.items()],
        *[(f"issues_{k}", v) for k, v in sorted(lv.items())],
        ("pii_removed", bool(args.drop_pii)),
        ("verify_sample_accidents", ver["accidents"] if ver else "not run"),
        ("verify_cells_rechecked", ver["cells"] if ver else "not run"),
        ("verify_cells_not_found_in_pdf", ver["not_found"] if ver else "not run"),
        ("null_rule", "Only NA / N/A / empty cells are blank; 'Not Known', 'Not Applicable' kept verbatim"),
        ("casualty_rule", "Blank cells in the persons table are recorded as 0 (the table's own totals confirm this)"),
    ]]
    order = ["Summary", "Accidents", "Vehicles", "Drivers", "Passengers", "Pedestrians", "Witnesses",
             "Transport_Inspections", "Road_Details", "Hospital_Details", "TMS_Details",
             "Validation_Issues", "Files_Log", "Data_Dictionary"]
    allsheets = dict(sheets); allsheets["Summary"] = summary; allsheets["Data_Dictionary"] = dd

    wb = Workbook(write_only=True)
    hdr_font = Font(bold=True, color="FFFFFF"); hdr_fill = PatternFill("solid", fgColor="1F4E78")
    from openpyxl.cell import WriteOnlyCell
    for name in order:
        data = allsheets.get(name, [])
        cols = []
        for r in data:
            for c in r:
                if c not in cols:
                    cols.append(c)
        ws = wb.create_sheet(title=name[:31])
        ws.freeze_panes = "B2" if name not in ("Summary", "Data_Dictionary") else "A2"
        for i, c in enumerate(cols, start=1):
            ws.column_dimensions[get_column_letter(i)].width = min(max(len(c) + 2, 12), 45)
        header = []
        for c in cols:
            cell = WriteOnlyCell(ws, value=c); cell.font = hdr_font; cell.fill = hdr_fill
            header.append(cell)
        ws.append(header)
        for r in data:
            out = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, str):
                    v = ILLEGAL_CHARACTERS_RE.sub("", v)
                    if len(v) > 32000:
                        v = v[:32000] + " …[truncated for Excel]"
                elif isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                out.append(v)
            ws.append(out)
        if cols:
            ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{max(len(data), 1) + 1}"
    os.makedirs(os.path.dirname(os.path.abspath(out_xlsx)), exist_ok=True)
    wb.save(out_xlsx)
    print(f"[ok] Excel written: {out_xlsx}")

    if args.parquet:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print("[warn] pyarrow not installed: skipping Parquet (pip install pyarrow)")
            args.parquet = False
    if args.parquet or args.csv:
        base = os.path.splitext(out_xlsx)[0]
        for name in order:
            df = pd.DataFrame(allsheets.get(name, []))
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].map(lambda x: x if x is None or isinstance(x, (str, int, float, bool, datetime))
                                      else str(x))
            if args.parquet:
                try:
                    df.astype({c: "string" for c in df.columns if df[c].dtype == object}).to_parquet(
                        f"{base}__{name}.parquet", index=False)
                except Exception as e:
                    print(f"[warn] parquet for {name} failed: {e}")
            if args.csv:
                df.to_csv(f"{base}__{name}.csv", index=False, encoding="utf-8-sig")
        print("[ok] Parquet/CSV copies written next to the workbook")

def verify_sample(sheets, n):
    """Independent self-check: extract each sampled PDF's plain text (not tables) and confirm that
    every verbatim value we stored for that accident occurs in it (whitespace-insensitive)."""
    import random
    random.seed(42)
    accs = sheets["Accidents"]
    sample = random.sample(accs, min(n, len(accs)))
    verbatim = {}
    for sec, (sheet, schema, _) in SECTIONS.items():
        verbatim.setdefault(sheet, set()).update(schema.values())
    verbatim["Accidents"].update(SUMMARY.values())
    by_acc = defaultdict(list)
    ids = {a["accident_id"] for a in sample}
    for sheet in ["Accidents", "Vehicles", "Drivers", "Passengers", "Pedestrians", "Witnesses",
                  "Transport_Inspections", "Road_Details", "Hospital_Details", "TMS_Details"]:
        for row in sheets[sheet]:
            if row["accident_id"] in ids:
                by_acc[row["accident_id"]].append((sheet, row))
    checked = missing = 0
    problems = []

    def occurs(value, tokens, joined):
        # contiguous match (whitespace-insensitive) OR the value's words appear in order with small gaps
        # (multi-line table cells are interleaved with neighbouring cells in plain-text extraction)
        if re.sub(r"\s", "", value) in joined:
            return True
        vt = value.split()
        starts = [i for i, t in enumerate(tokens) if vt and vt[0] in t]
        for st in starts:
            pos, ok = st, True
            for w in vt[1:]:
                nxt = next((j for j in range(pos + 1, min(pos + 40, len(tokens))) if w in tokens[j]), None)
                if nxt is None:
                    ok = False
                    break
                pos = nxt
            if ok:
                return True
        return False

    for a in sample:
        with pdfplumber.open(a["source_file"]) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        tokens = text.split()
        joined = re.sub(r"\s", "", text)
        for sheet, row in by_acc[a["accident_id"]]:
            for c, v in row.items():
                if c not in verbatim.get(sheet, ()) or v is None or not isinstance(v, str):
                    continue
                checked += 1
                if not occurs(v, tokens, joined):
                    missing += 1
                    problems.append(OrderedDict(level="error", accident_id=a["accident_id"], sheet=sheet, field=c,
                                                value=v[:500], issue="VERIFY: value not found in PDF text",
                                                source_file=a["source_file"]))
    sheets["Validation_Issues"].extend(problems)
    print(f"[verify] {len(sample)} random accidents, {checked} cells re-checked against PDF text, {missing} not found")
    return {"accidents": len(sample), "cells": checked, "not_found": missing}

# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder containing the iRAD PDFs (searched recursively)")
    ap.add_argument("output", help="output .xlsx path")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--drop-pii", action="store_true")
    ap.add_argument("--parquet", action="store_true")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=200,
                    help="re-check N random accidents: every verbatim cell must occur in the PDF text (0 = off)")
    args = ap.parse_args()

    pdfs = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    pdfs.sort()
    if args.limit:
        pdfs = pdfs[:args.limit]
    print(f"[info] {len(pdfs)} PDF files found under {args.input}")

    cache_path = args.output + ".cache.jsonl"
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("parser_version") == PARSER_VERSION:
                        done[r["source_file"]] = r
                except json.JSONDecodeError:
                    pass
        print(f"[info] resuming: {len(done)} files already parsed (cache {cache_path})")
    todo = [p for p in pdfs if p not in done]
    t0 = time.time()
    if todo:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as cache, Pool(args.workers) as pool:
            for k, rec in enumerate(pool.imap_unordered(file_worker, todo, chunksize=8), start=1):
                cache.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                done[rec["source_file"]] = rec
                if k % 200 == 0 or k == len(todo):
                    el = time.time() - t0
                    rate = k / el if el else 0
                    eta = (len(todo) - k) / rate / 60 if rate else 0
                    print(f"[parse] {k}/{len(todo)}  {rate:.1f} files/s  ETA {eta:.1f} min", flush=True)
                    cache.flush()
    records = [done[p] for p in pdfs if p in done]
    # json round-trip turns defaultdicts into dicts already; normalise entity dicts
    for r in records:
        if isinstance(r.get("entities"), dict):
            r["entities"] = {k: [OrderedDict(e) for e in v] for k, v in r["entities"].items()}
    print("[info] building tables and running validation checks ...")
    sheets = build(records, drop_pii=args.drop_pii)
    if args.verify:
        sheets["_verify"] = verify_sample(sheets, args.verify)
    write_outputs(sheets, args.output, args, len(pdfs))
    lv = Counter(i["level"] for i in sheets["Validation_Issues"])
    print(f"[done] accidents={len(sheets['Accidents'])} vehicles={len(sheets['Vehicles'])} "
          f"drivers={len(sheets['Drivers'])} passengers={len(sheets['Passengers'])} "
          f"pedestrians={len(sheets['Pedestrians'])}  issues={dict(lv)}  "
          f"time={(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()
