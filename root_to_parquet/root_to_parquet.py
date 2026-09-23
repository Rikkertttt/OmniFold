import os
import time
import read_data as rd
import event_selection_filters as filter
from collections import defaultdict
from tqdm import tqdm

total_reco_counts = defaultdict(int)
total_gen_counts = defaultdict(int)

DIR = "/dcache/atlas/llehmann/Eventgeneration/standalone/hww/"
files = sorted([f for f in os.listdir(DIR) if f.endswith(".root")])

save_DIR = "/data/atlas/users/rvanrhee/hww_parquet/"
os.makedirs(save_DIR + "reco/", exist_ok=True)
os.makedirs(save_DIR + "gen/",  exist_ok=True)

for file in tqdm(files, desc="Processing files", unit="file"):
    t0 = time.time()

    PATH = DIR + file + ":Delphes"
    tree = rd.open_root_file(PATH, show_keys=False, verbose=False)

    reco_event = rd.load_reco_objects(tree, verbose=False)
    gen_event  = rd.load_gen_objects(tree, verbose=False)

    reco_event, counts = filter.full_preselection(reco_event)
    reco_event, counts = filter.full_selection_reco(reco_event, counts=counts)
    for label, n in counts.items():
        total_reco_counts[label] += n

    gen_event = rd.EventObjects(
        event_id  = gen_event.event_id,
        jets      = gen_event.jets,
        electrons = gen_event.electrons[gen_event.electrons.pt > 15],
        muons     = gen_event.muons[gen_event.muons.pt > 15],
        met       = gen_event.met,
    )

    gen_event, counts = filter.full_preselection(gen_event)
    gen_event, counts = filter.full_selection_gen(gen_event, counts=counts)
    for label, n in counts.items():
        total_gen_counts[label] += n

    stem = os.path.splitext(file)[0]
    rd.save_events(reco_event, f"{save_DIR}reco/reco_{stem}.parquet")
    rd.save_events(gen_event,  f"{save_DIR}gen/gen_{stem}.parquet")

    tqdm.write(f"  {file} done in {time.time() - t0:.1f}s")

def write_counts(counts: dict, path: str) -> None:
    with open(path, "w") as f:
        for label, n in counts.items():
            f.write(f"{label}:\t{n}\n")

write_counts(total_reco_counts, f"{save_DIR}reco/counts.txt")
write_counts(total_gen_counts,  f"{save_DIR}gen/counts.txt")

print("\nReco -----------------------")
for label, n in total_reco_counts.items():
    print(f"N after {label}:\t\t{n}")

print("\nGen -----------------------")
for label, n in total_gen_counts.items():
    print(f"N after {label}:\t\t{n}")