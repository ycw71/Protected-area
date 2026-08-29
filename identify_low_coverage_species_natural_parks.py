import math
import os
from collections import defaultdict

import arcpy

arcpy.env.overwriteOutput = True

# Input data -----------------------------------------------------------

# Final IUCN species distribution layer. Use one group at a time.
# Mammals:    ./data/IUCN_mammals_final.shp
# Amphibians: ./data/IUCN_amphibians_final.shp
# Reptiles:   ./data/IUCN_reptiles_final.shp
species_fc = r"./data/IUCN_mammals_final.shp"
taxon = "mammals"

# Existing nature reserves, candidate natural parks, and study-area boundary.
existing_reserves = r"./data/existing_nature_reserves.shp"
candidate_natural_parks = r"./data/candidate_natural_parks.shp"
study_area = r"./data/study_area.shp"

out_gdb = r"./output/results.gdb"

# Parameters -----------------------------------------------------------

categories = ["VU", "EN", "CR", "NT", "LC"]
rank_threshold = 0.35
select_smallest_area = True

species_id_field = "id_no"
species_name_field = "sci_name"
category_field = "category"
park_id_field = "gml_id"

# Prepare spatial reference -------------------------------------------

target_sr = arcpy.Describe(existing_reserves).spatialReference

species_projected = os.path.join(out_gdb, f"{taxon}_projected")
species_sr = arcpy.Describe(species_fc).spatialReference

if species_sr.name != target_sr.name:
    if arcpy.Exists(species_projected):
        arcpy.management.Delete(species_projected)
    arcpy.management.Project(species_fc, species_projected, target_sr)
    species_use = species_projected
else:
    species_use = species_fc

parks_projected = os.path.join(out_gdb, "candidate_natural_parks_projected")
parks_sr = arcpy.Describe(candidate_natural_parks).spatialReference

if parks_sr.name != target_sr.name:
    if arcpy.Exists(parks_projected):
        arcpy.management.Delete(parks_projected)
    arcpy.management.Project(candidate_natural_parks, parks_projected, target_sr)
    parks_use = parks_projected
else:
    parks_use = candidate_natural_parks

# Select species occurring in the study area --------------------------

all_species_layer = f"{taxon}_all_species_lyr"
arcpy.management.MakeFeatureLayer(
    species_use,
    all_species_layer,
    f"{category_field} IN ('VU', 'EN', 'CR', 'NT', 'LC')",
)

arcpy.management.SelectLayerByLocation(
    all_species_layer,
    overlap_type="INTERSECT",
    select_features=study_area,
    selection_type="NEW_SELECTION",
)

study_species_ids = set()

with arcpy.da.SearchCursor(all_species_layer, [species_id_field]) as cursor:
    for (species_id,) in cursor:
        if species_id not in (None, "", " "):
            study_species_ids.add(species_id)

if not study_species_ids:
    raise RuntimeError("No species intersect the study area.")

arcpy.management.SelectLayerByAttribute(
    all_species_layer,
    "CLEAR_SELECTION",
)

id_field_type = next(
    f.type
    for f in arcpy.ListFields(species_use)
    if f.name == species_id_field
)


def make_in_sql(dataset, field_name, values, field_type):
    field = arcpy.AddFieldDelimiters(dataset, field_name)

    if field_type in ("String", "Guid"):
        values_text = ",".join(f"'{str(v)}'" for v in values)
    else:
        values_text = ",".join(str(v) for v in values)

    return f"{field} IN ({values_text})"


study_species_layer = f"{taxon}_study_species_lyr"

arcpy.management.MakeFeatureLayer(
    species_use,
    study_species_layer,
    f"{category_field} IN ('VU', 'EN', 'CR', 'NT', 'LC')",
)

arcpy.management.SelectLayerByAttribute(
    study_species_layer,
    "CLEAR_SELECTION",
)

study_species_ids = list(study_species_ids)
chunk_size = 900

for i in range(0, len(study_species_ids), chunk_size):
    chunk = study_species_ids[i:i + chunk_size]

    sql = make_in_sql(
        species_use,
        species_id_field,
        chunk,
        id_field_type,
    )

    arcpy.management.SelectLayerByAttribute(
        study_species_layer,
        "ADD_TO_SELECTION",
        sql,
    )

study_species_fc = os.path.join(
    out_gdb,
    f"{taxon}_study_area_species",
)

if arcpy.Exists(study_species_fc):
    arcpy.management.Delete(study_species_fc)

arcpy.management.CopyFeatures(
    study_species_layer,
    study_species_fc,
)

# Calculate overlap with existing nature reserves ---------------------

covered_intersection = os.path.join(
    out_gdb,
    f"{taxon}_existing_reserve_intersection",
)

if arcpy.Exists(covered_intersection):
    arcpy.management.Delete(covered_intersection)

arcpy.analysis.PairwiseIntersect(
    in_features=[study_species_fc, existing_reserves],
    out_feature_class=covered_intersection,
    join_attributes="ALL",
)

species_info = {}

with arcpy.da.SearchCursor(
    study_species_fc,
    [species_id_field, species_name_field, category_field],
) as cursor:
    for species_id, species_name, category in cursor:
        if species_id not in (None, "", " "):
            species_info[species_id] = {
                "sci_name": species_name,
                "category": category,
            }

species_old_pa_area = defaultdict(float)

with arcpy.da.SearchCursor(
    covered_intersection,
    [species_id_field, "SHAPE@AREA"],
) as cursor:
    for species_id, area in cursor:
        if species_id not in (None, "", " "):
            species_old_pa_area[species_id] += area

species_area_list = []

for species_id, info in species_info.items():
    species_area_list.append(
        {
            "id_no": species_id,
            "sci_name": info["sci_name"],
            "category": info["category"],
            "old_pa_area": species_old_pa_area.get(species_id, 0.0),
        }
    )

# Rank species by current overlap ----------------------

species_area_list.sort(
    key=lambda x: x["old_pa_area"],
    reverse=not select_smallest_area,
)

total_species_count = len(species_area_list)

selected_species_count = max(
    1,
    math.ceil(total_species_count * rank_threshold),
)

selected_species = species_area_list[:selected_species_count]
selected_ids = {item["id_no"] for item in selected_species}

ranked_species = os.path.join(
    out_gdb,
    f"{taxon}_rank{int(rank_threshold * 100)}pct_ranked",
)

if arcpy.Exists(ranked_species):
    arcpy.management.Delete(ranked_species)

arcpy.management.CopyFeatures(
    study_species_fc,
    ranked_species,
)

existing_fields = {f.name for f in arcpy.ListFields(ranked_species)}

if "oldpa_area" not in existing_fields:
    arcpy.management.AddField(ranked_species, "oldpa_area", "DOUBLE")

if "oldpa_rank" not in existing_fields:
    arcpy.management.AddField(ranked_species, "oldpa_rank", "LONG")

if "oldpa_pct" not in existing_fields:
    arcpy.management.AddField(ranked_species, "oldpa_pct", "DOUBLE")

if "select_gnn" not in existing_fields:
    arcpy.management.AddField(ranked_species, "select_gnn", "SHORT")

rank_dict = {}
pct_dict = {}

for rank, item in enumerate(species_area_list, start=1):
    species_id = item["id_no"]
    rank_dict[species_id] = rank
    pct_dict[species_id] = rank / total_species_count

with arcpy.da.UpdateCursor(
    ranked_species,
    [
        species_id_field,
        "oldpa_area",
        "oldpa_rank",
        "oldpa_pct",
        "select_gnn",
    ],
) as cursor:
    for row in cursor:
        species_id = row[0]
        row[1] = species_old_pa_area.get(species_id, 0.0)
        row[2] = rank_dict.get(species_id)
        row[3] = pct_dict.get(species_id)
        row[4] = 1 if species_id in selected_ids else 0
        cursor.updateRow(row)

# Intersect selected species with candidate natural parks -------------

for category in categories:
    selected_layer = f"{taxon}_{category}_selected_lyr"

    arcpy.management.MakeFeatureLayer(
        ranked_species,
        selected_layer,
        f"{category_field} = '{category}' AND select_gnn = 1",
    )

    selected_species_fc = os.path.join(
        out_gdb,
        f"{taxon}_{category}_rank{int(rank_threshold * 100)}pct_species",
    )

    if arcpy.Exists(selected_species_fc):
        arcpy.management.Delete(selected_species_fc)

    arcpy.management.CopyFeatures(
        selected_layer,
        selected_species_fc,
    )

    final_result = os.path.join(
        out_gdb,
        f"{taxon}_{category}_rank{int(rank_threshold * 100)}pct_in_natural_parks",
    )

    if arcpy.Exists(final_result):
        arcpy.management.Delete(final_result)

    arcpy.analysis.PairwiseIntersect(
        in_features=[selected_species_fc, parks_use],
        out_feature_class=final_result,
        join_attributes="ALL",
    )
