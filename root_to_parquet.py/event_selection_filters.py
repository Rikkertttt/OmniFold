import awkward as ak
import numpy as np
import vector

from read_data import EventObjects


def apply_event_mask(events: EventObjects,
                    event_mask: ak.Array) -> EventObjects:
    """
    Apply an event-level mask to every event-aligned collection.
    """

    return EventObjects(
        event_id=events.event_id[event_mask],
        jets=events.jets[event_mask],
        muons=events.muons[event_mask],
        electrons=events.electrons[event_mask],
        met=events.met[event_mask],
    )

def jet_pt_eta_mask(events: EventObjects, min_pt: float, max_abs_eta: float) -> EventObjects:
    """
    Keep jets with pt above min_pt and absolute eta below max_abs_eta.
    """
    jet_pt_mask = events.jets.pt > min_pt
    jet_eta_mask = abs(events.jets.eta) < max_abs_eta
    total_jet_mask = jet_pt_mask & jet_eta_mask

    return EventObjects(
            event_id=events.event_id,
            jets=events.jets[total_jet_mask],
            muons=events.muons,
            electrons=events.electrons,
            met=events.met,
        )



def has_atleast_1_muon(events: EventObjects, prints=False) -> EventObjects:
    """
    Keep events containing at least one muon.
    """

    has_muons_mask = ak.num(events.muons) >= 1

    selected_events = apply_event_mask(events, has_muons_mask)

    if prints:
        print(f"N of samples after muon number filter:\t\t {len(selected_events.jets)}")

    return selected_events

def has_atleast_1_electron(events: EventObjects, prints=False) -> EventObjects:
    """
    Keep events containing at least one electron.
    """
    has_electrons_mask = ak.num(events.electrons) >= 1

    selected_events = apply_event_mask(events, has_electrons_mask)

    if prints:
        print(f"N of samples after electron number filter:\t {len(selected_events.jets)}")

    return selected_events

def order_objects_on_pt(events: EventObjects) -> EventObjects:
    """
    Sort jets, muons, and electrons independently by descending pt within every event.
    """

    jet_order = ak.argsort(
        events.jets.pt,
        axis=1,
        ascending=False,
    )

    muon_order = ak.argsort(
        events.muons.pt,
        axis=1,
        ascending=False,
    )

    electron_order = ak.argsort(
        events.electrons.pt,
        axis=1,
        ascending=False,
    )

    return EventObjects(
        event_id=events.event_id,
        jets=events.jets[jet_order],
        muons=events.muons[muon_order],
        electrons=events.electrons[electron_order],
        met=events.met,
    )


def leptons_have_opposite_charge(events: EventObjects, prints=False) -> EventObjects:
    """
    Require the leading muon and leading electron to have opposite charge.
    """

    leading_muon = events.muons[:, 0]
    leading_electron = events.electrons[:, 0]

    charge_mask = leading_muon.charge == -leading_electron.charge

    selected_events = apply_event_mask(events, charge_mask)

    if prints:
        print(f"N of samples after opposite charge filter:\t {len(selected_events.jets)}")

    return selected_events


def lepton_pt_mask(events: EventObjects,
                   leading_pt: float, 
                   subleading_pt: float, 
                   prints = False) -> EventObjects:
    """
    Require the two highest-pt leptons in each event to pass pt thresholds.
    """

    # Combine leptons
    leptons = ak.concatenate([events.electrons, events.muons], axis=1)

    # Sort leptons by descending energy
    leptons_sorted = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]

    # Extract leading and subleading leptons
    leading = leptons_sorted[:, 0]
    subleading = leptons_sorted[:, 1]

    # Create masks based on energy thresholds
    leading_mask = leading.pt >= leading_pt
    subleading_mask = subleading.pt >= subleading_pt

    # Combine masks
    total_mask = leading_mask & subleading_mask

    selected_event = apply_event_mask(events, total_mask)

    if prints:
        print(f"N of samples after lepton pt filter:\t\t {len(selected_event.jets)}")

    return selected_event



def lepton_eta_mask(events: EventObjects,
                    electron_eta: float, 
                    muon_eta: float, 
                    prints=False) -> EventObjects:
    """
    Require the leading electron and leading muon to pass eta acceptance cuts.
    """

    leading_muon = events.muons[:, 0]
    leading_electron = events.electrons[:, 0]

    m_eta_mask = abs(leading_muon.eta) < muon_eta
    e_eta_mask = abs(leading_electron.eta) < electron_eta
    total_mask = m_eta_mask & e_eta_mask

    selected_events = apply_event_mask(events, total_mask)

    if prints:
        print(f"N of samples after lepton eta filter:\t\t {len(selected_events.jets)}")

    return selected_events

def leptons_deltaR_mask(events: EventObjects, 
                        deltaR: float, 
                        prints: bool = False) -> EventObjects:
    """
    Require the highest-pt electron and highest-pt muon to be separated
    by more than deltaR.
    """

    leading_muon = events.muons[:, 0]
    leading_electron = events.electrons[:, 0]

    deltaR_mask = (
        leading_muon.deltaR(leading_electron) > deltaR
    )

    selected_events = apply_event_mask(events, deltaR_mask)

    if prints:
        print(
            "N of samples after lepton deltaR filter:\t "
            f"{len(selected_events.jets)}"
        )

    return selected_events

def lepton_masses_mask(events: EventObjects, mass_cutoff: float, prints=False) -> EventObjects:
    """
    Require the leading electron-muon pair mass to exceed mass_cutoff.
    """

    leading_muon = events.muons[:, 0]
    leading_electron = events.electrons[:, 0]

    mass_mask = (leading_muon + leading_electron).mass > mass_cutoff

    selected_events = apply_event_mask(events, mass_mask)

    if prints:
        print(f"N of samples after lepton mass filter:\t\t {len(selected_events.jets)}")

    return selected_events

def lepton_jet_deltaR_mask(events: EventObjects,
                           deltaR_thres: float,
                           prints: bool = False) -> EventObjects:
    """
    Require both leading jets to be separated from both leading leptons.
    """
 
    leading_jets = events.jets[:, :2]
    leading_leptons = ak.concatenate(
        [
            events.muons[:, :1],
            events.electrons[:, :1],
        ],
        axis=1,
    )

    # Four pairs per event:
    # jet 1–muon, jet 1–electron, jet 2–muon, jet 2–electron.
    pairs = ak.cartesian(
        {
            "jet": leading_jets,
            "lepton": leading_leptons,
        },
        axis=1,
    )

    delta_r = pairs["jet"].deltaR(pairs["lepton"])

    # Keep the event only when every selected jet-lepton pair is separated.
    event_mask = ak.all(delta_r > deltaR_thres, axis=1)

    selected_events = apply_event_mask(events, event_mask)

    if prints:
        print(
            "N of samples after lepton-jet deltaR filter:\t "
            f"{len(selected_events.jets)}"
        )

    return selected_events

def has_at_least_two_jets(events: EventObjects, prints=False) -> EventObjects:
    """
    Keep events containing at least two retained jets.
    """

    two_or_more_mask = ak.num(events.jets, axis=-1) >= 2

    selected_events = apply_event_mask(events, two_or_more_mask)

    if prints:
        print(f"N of samples number of jets filter:\t\t {len(selected_events.jets)}")

    return selected_events

def remove_btag(events: EventObjects, pt_thres: float, eta_thres: float, prints=False) -> EventObjects:
    """
    Reject events containing a b-tagged jet within the specified acceptance.
    """

    b_tagged = events.jets.btag == 1
    pt_mask = events.jets.pt > pt_thres
    eta_mask = abs(events.jets.eta) < eta_thres

    total_mask = ~ak.any(((b_tagged & pt_mask) & eta_mask), axis=1)

    selected_events = apply_event_mask(events, total_mask)

    if prints:
        print(f"N of samples after btag removal:\t\t {len(selected_events.jets)}")

    return selected_events

def central_jet_veto(events: EventObjects, pt_cutoff: float, prints=False):
    """
    Reject events with an additional central jet above pt_cutoff.
    """

    leading_jet = events.jets[:, 0]
    subleading_jet = events.jets[:, 1]

    additional_jets = events.jets[:, 2:]

    rapidity_min = np.minimum(
        leading_jet.rapidity,
        subleading_jet.rapidity
    )
    rapidity_max = np.maximum(
        leading_jet.rapidity,
        subleading_jet.rapidity
    )

    central_additional_jet = (
        (additional_jets.pt > pt_cutoff)
        & (additional_jets.rapidity > rapidity_min)
        & (additional_jets.rapidity < rapidity_max)
    )

    CJV_mask = ~ak.any(central_additional_jet, axis=1)

    selected_events = apply_event_mask(events, CJV_mask)

    if prints:
        print(f"N of samples after CJV mask:\t\t\t {len(selected_events.jets)}")

    return selected_events

def outside_lepton_veto(events: EventObjects, prints=False):
    """
    Require the leading electron and muon to lie between the two leading jets.
    """

    leading_jet = events.jets[:, 0]
    subleading_jet = events.jets[:, 1]

    leading_muon = events.muons[:, 0]
    leading_electron = events.electrons[:, 0]

    rapidity_min = np.minimum(
        leading_jet.rapidity,
        subleading_jet.rapidity,
    )
    rapidity_max = np.maximum(
        leading_jet.rapidity,
        subleading_jet.rapidity,
    )

    muon_outside = (
        (leading_muon.rapidity < rapidity_min)
        | (leading_muon.rapidity > rapidity_max)
    )

    electron_outside = (
        (leading_electron.rapidity < rapidity_min)
        | (leading_electron.rapidity > rapidity_max)
    )

    # Keep events only if both selected leptons are inside the interval.
    event_mask = ~(muon_outside | electron_outside)

    selected_events = apply_event_mask(events, event_mask)


    if prints:
        print(f"N of samples after OLV mask:\t\t\t {len(selected_events.jets)}")

    return selected_events

def jets_mass_mask(events: EventObjects, mass_cutoff: float, prints=False):
    """
    Require the invariant mass of the two leading jets to exceed mass_cutoff.
    """

    mass_mask = (events.jets[:, 0] + events.jets[:, 1]).mass > mass_cutoff

    selected_events = apply_event_mask(events, mass_mask)

    if prints:
        print(f"N of samples after jet mass filter: \t\t {len(selected_events.jets)}")

    return selected_events

def jet_rapidity_mask(events: EventObjects, rapidity_diff_cutoff: float, prints=False):
    """
    Require the two leading jets to have a minimum rapidity separation.
    """

    delta_y = abs(events.jets[:, 0].rapidity - events.jets[:, 1].rapidity)
    delta_y_mask = delta_y >= rapidity_diff_cutoff

    selected_events = apply_event_mask(events, delta_y_mask)

    if prints:
        print(f"N of samples after jet delta-y filter: \t\t {len(selected_events.jets)}")

    return selected_events

def lepton_phi_mask(events: EventObjects, delta_phi_cutoff: float, prints=False):
    """
    Require the leading electron-muon pair to have delta-phi below the cutoff.
    """

    delta_phi = abs((events.muons[:, 0].phi - events.electrons[:, 0].phi + np.pi) % (2*np.pi) - np.pi)

    delta_phi_mask = delta_phi <= delta_phi_cutoff

    selected_events = apply_event_mask(events, delta_phi_mask)

    if prints:
        print(f"N of samples after lepton delta-phi filter: \t {len(selected_events.jets)}")

    return selected_events

def full_preselection(events: EventObjects, prints=False) -> EventObjects:
    """
    Apply baseline jet, lepton, and b-tag preselection requirements.
    """

    if prints:
        print(f"N of samples at start:\t\t\t\t {len(events.jets)}")

    events = order_objects_on_pt(events)

    events = jet_pt_eta_mask(events, min_pt=20, max_abs_eta=4.5)
    events = has_atleast_1_muon(events, prints=prints)
    events = has_atleast_1_electron(events, prints=prints)
    events = leptons_have_opposite_charge(events, prints=prints)
    events = lepton_pt_mask(events, leading_pt=22, subleading_pt=15, prints=prints)
    events = lepton_eta_mask(events, electron_eta=2.5, muon_eta=2.5, prints=prints)
    events = leptons_deltaR_mask(events, deltaR=0.1, prints=prints)
    events = lepton_masses_mask(events, mass_cutoff=10, prints=prints)
    events = lepton_jet_deltaR_mask(events, deltaR_thres=0.4, prints=prints)
    events = has_at_least_two_jets(events, prints=prints)
    events = remove_btag(events, pt_thres=20, eta_thres=2.5, prints=prints)     

    return events

def full_selection(events: EventObjects, prints=False) -> EventObjects:
    """
    Apply VBF topology and dilepton angular selection requirements.
    """

    # events = central_jet_veto(events, pt_cutoff = 20, prints=prints)
    events = outside_lepton_veto(events, prints=prints)
    events = jets_mass_mask(events, mass_cutoff=450, prints=prints)
    events = jet_rapidity_mask(events, rapidity_diff_cutoff=2.1, prints=prints)
    events = lepton_phi_mask(events, delta_phi_cutoff=1.4, prints=prints)

    return events