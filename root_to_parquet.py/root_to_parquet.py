import read_data as rd
import awkward as ak
import plotting_features as plotting
import event_selection_filters as filter

PATH="/dcache/atlas/llehmann/Eventgeneration/standalone/hww/delphes.hww.10.root:Delphes"

tree = rd.open_root_file(PATH, show_keys=False, verbose=False)

reco_event = rd.load_reco_objects(tree, verbose=False)
gen_event = rd.load_gen_objects(tree, verbose=False)

reco_event = filter.full_preselection(reco_event, prints=True)
reco_event = filter.full_selection(reco_event, prints=True)

gen_event = rd.EventObjects(
    event_id = gen_event.event_id,
    jets = gen_event.jets,
    electrons = gen_event.electrons[gen_event.electrons.pt > 15],
    muons = gen_event.muons[gen_event.muons.pt > 15],
    met = gen_event.met
)

gen_event = filter.full_preselection(gen_event, prints=True)
gen_event = filter.full_selection(gen_event, prints=True)

