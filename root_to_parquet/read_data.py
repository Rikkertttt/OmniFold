from typing import NamedTuple

import awkward as ak
import numpy as np
import uproot
import vector

vector.register_awkward()

MUON_MASS_GEV = 0.105658
ELECTRON_MASS_GEV = 0.000511


class EventObjects(NamedTuple):
    """Collections aligned event-by-event through event_id."""

    event_id: np.ndarray
    jets: ak.Array
    muons: ak.Array
    electrons: ak.Array
    met: ak.Array


def open_root_file(path: str, show_keys: bool = False, verbose: bool = False):
    """Open a ROOT file and optionally print its top-level keys."""
    if verbose:
        print(f"Opening {path}")

    root_file = uproot.open(path)

    if show_keys:
        print(root_file.keys())

    return root_file


def make_four_momentum(
    pt: ak.Array,
    eta: ak.Array,
    phi: ak.Array,
    mass: ak.Array,
) -> ak.Array:
    """Construct an Awkward Momentum4D array from pt, eta, phi, and mass."""
    return ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": mass,
        },
        with_name="Momentum4D",
    )


def load_reco_objects(tree, verbose: bool = False) -> EventObjects:
    """Load reconstructed objects with charges and b-tags attached.

    Jet fields:
        pt, eta, phi, mass, btag

    Muon/electron fields:
        pt, eta, phi, mass, charge
    """

    branches = [
        "Jet.PT",
        "Jet.Eta",
        "Jet.Phi",
        "Jet.Mass",
        "Jet.BTag",
        "Muon.PT",
        "Muon.Eta",
        "Muon.Phi",
        "Muon.Charge",
        "Electron.PT",
        "Electron.Eta",
        "Electron.Phi",
        "Electron.Charge",
        "MissingET.MET",
        "MissingET.Phi",
    ]

    if verbose: print("Loading reconstructed objects")

    arrays = tree.arrays(branches, library="ak")

    # Original TTree entry number: preserve this during event-level filtering.
    event_id = np.arange(tree.num_entries)

    jets = make_four_momentum(
        arrays["Jet.PT"],
        arrays["Jet.Eta"],
        arrays["Jet.Phi"],
        arrays["Jet.Mass"],
    )
    jets = ak.with_field(jets, arrays["Jet.BTag"], "btag")
    if verbose: print("Made Jets 4-momentum")

    muons = make_four_momentum(
        arrays["Muon.PT"],
        arrays["Muon.Eta"],
        arrays["Muon.Phi"],
        ak.full_like(arrays["Muon.PT"], MUON_MASS_GEV),
    )
    muons = ak.with_field(muons, arrays["Muon.Charge"], "charge")
    if verbose: print("Made Muons 4-momentum")

    electrons = make_four_momentum(
        arrays["Electron.PT"],
        arrays["Electron.Eta"],
        arrays["Electron.Phi"],
        ak.full_like(arrays["Electron.PT"], ELECTRON_MASS_GEV),
    )
    electrons = ak.with_field(
        electrons,
        arrays["Electron.Charge"],
        "charge",
    )
    if verbose: print("Made Electrons 4-momentum")

    met_pt = arrays["MissingET.MET"]
    met = make_four_momentum(
        pt=met_pt,
        eta=ak.zeros_like(met_pt),
        phi=arrays["MissingET.Phi"],
        mass=ak.zeros_like(met_pt),
    )
    if verbose: print("Made MET 4-momentum")

    return EventObjects(
        event_id=event_id,
        jets=jets,
        muons=muons,
        electrons=electrons,
        met=met
    )


def load_gen_objects(tree, verbose: bool = False) -> EventObjects:
    """Load generator-level jets, electrons, and muons.

    Both charge signs are selected:
        - electron/muon: PDG ID 11/13, charge -1;
        - positron/antimuon: PDG ID -11/-13, charge +1.

    Assumes that the Delphes tree contains GenJet.BTag.
    """
    branches = [
        "GenJet.PT",
        "GenJet.Eta",
        "GenJet.Phi",
        "GenJet.Mass",
        "GenJet.BTag",
        "Particle.PID",
        "Particle.PT",
        "Particle.Eta",
        "Particle.Phi",
        "Particle.Charge",
        "GenMissingET.MET",
        "GenMissingET.Phi",
    ]

    if verbose: print("Loading generator-level objects")

    arrays = tree.arrays(branches, library="ak")
    event_id = np.arange(tree.num_entries)

    gen_jets = make_four_momentum(
        arrays["GenJet.PT"],
        arrays["GenJet.Eta"],
        arrays["GenJet.Phi"],
        arrays["GenJet.Mass"],
    )
    gen_jets = ak.with_field(
        gen_jets,
        arrays["GenJet.BTag"],
        "btag",
    )
    if verbose: print("Made Jets 4-momentum")

    pid = arrays["Particle.PID"]

    electron_mask = abs(pid) == 11
    muon_mask = abs(pid) == 13

    gen_muons = make_four_momentum(
        arrays["Particle.PT"][muon_mask],
        arrays["Particle.Eta"][muon_mask],
        arrays["Particle.Phi"][muon_mask],
        ak.full_like(
            arrays["Particle.PT"][muon_mask],
            MUON_MASS_GEV,
        ),
    )
    gen_muons = ak.with_field(
        gen_muons,
        arrays["Particle.Charge"][muon_mask],
        "charge",
    )
    if verbose: print("Made Muons 4-momentum")

    gen_electrons = make_four_momentum(
        arrays["Particle.PT"][electron_mask],
        arrays["Particle.Eta"][electron_mask],
        arrays["Particle.Phi"][electron_mask],
        ak.full_like(
            arrays["Particle.PT"][electron_mask],
            ELECTRON_MASS_GEV,
        ),
    )
    gen_electrons = ak.with_field(
        gen_electrons,
        arrays["Particle.Charge"][electron_mask],
        "charge",
    )
    if verbose: print("Made Electrons 4-momentum")

    genmet_pt = arrays["GenMissingET.MET"]

    gen_met = make_four_momentum(
        pt=genmet_pt,
        eta=ak.zeros_like(genmet_pt),
        phi=arrays["GenMissingET.Phi"],
        mass=ak.zeros_like(genmet_pt),
    )
    if verbose: print("Made MET 4-momentum")

    return EventObjects(
        event_id=event_id,
        jets=gen_jets,
        muons=gen_muons,
        electrons=gen_electrons,
        met=gen_met
    )

def save_events(events: EventObjects, path: str) -> None:
    """Save an EventObjects to a parquet file."""
    ak.to_parquet(
        ak.zip({
            "event_id": ak.from_regular(events.event_id[:, np.newaxis])[:, 0],
            "jets":      events.jets,
            "muons":     events.muons,
            "electrons": events.electrons,
            "met":       events.met,
        }),
        path,
    )

def load_events(path: str) -> EventObjects:
    """Load an EventObjects from a parquet file."""
    array = ak.from_parquet(path)
    return EventObjects(
        event_id=np.asarray(array["event_id"]),
        jets=array["jets"],
        muons=array["muons"],
        electrons=array["electrons"],
        met=array["met"],
    )