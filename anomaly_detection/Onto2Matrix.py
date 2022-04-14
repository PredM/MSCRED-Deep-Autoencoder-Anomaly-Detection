import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import owlready2 as owl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from framework.Dataset import Dataset
#from analytic_tools.TSNE_Vis import discrete_cmap
from configuration.Configuration import Configuration


def plot(feature_names, linked_features, responsible_relations, force_self_loops, display_labels):
    # Recreate the ADJ but with different values = colors based on the relationship of the connection
    n = feature_names.size
    a_plot = pd.DataFrame(index=feature_names, columns=feature_names, data=np.zeros(shape=(n, n)))

    color_values = {
        'no_relation': 0,
        'self_loops': 1,
        'component': 2,
        'same_iri': 3,
        'connection': 4,
        'actuation': 5,
        'calibration': 6,
        'precondition': 7,
        'postcondition': 8,
    }

    if force_self_loops:

        for f_j in feature_names:
            a_plot.loc[f_j, f_j] = color_values['self_loops']

    for (f_j, f_i), r in zip(linked_features, responsible_relations):
        c_val = color_values[r]

        if c_val > a_plot.loc[f_i, f_j]:
            a_plot.loc[f_i, f_j] = c_val

    size = 22 if display_labels else 15
    dpi = 200 if display_labels else 300

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=dpi)
    im = ax.imshow(a_plot.values, interpolation='none', cmap=discrete_cmap(len(list(color_values.keys())), 'jet'), )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    im.set_clim(vmin=-0.5, vmax=len(list(color_values.keys())) - 0.5)
    ax.set_title(color_values)

    ax.set_ylabel('i (target)')
    ax.set_xlabel('j (source)')

    ax.tick_params(which='minor', width=0)
    ax.set_xticks(np.arange(-.5, n, 10), minor=True)
    ax.set_yticks(np.arange(-.5, n, 10), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

    if display_labels:
        # Minor ticks with width = 0 so they are not really visible
        ax.set_xticks(np.arange(0, n, 1), minor=False)
        ax.set_yticks(np.arange(0, n, 1), minor=False)

        features = [f[0:20] if len(f) > 20 else f for f in a_plot.columns]

        ax.set_xticklabels(features, minor=False)
        ax.set_yticklabels(features, minor=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(f"../logs/{config.a_pre_file.split('.')[0]}.pdf", dpi=dpi, bbox_inches='tight')

    plt.show()


# Split string at pos th occurrence of sep
def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def name_2_iri(feature_names: np.ndarray):
    """
    Creates a mapping from feature names to the iris in the ontology.
    :param feature_names: A numpy array with features names that matches the order of features in the dataset.
    :return: A dictionary that matches the feature names to their iri.
    """

    base_iri = 'FTOnto:'
    feature_2_iri = {}
    manual_corrections = {
        base_iri + "OV_1_Compressor_8": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "WT_1_Compressor_8": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "OV_2_Compressor_8": base_iri + "OV_2_WT_2_Compressor_8",
        base_iri + "WT_2_Compressor_8": base_iri + "OV_2_WT_2_Compressor_8",
        base_iri + "MM_1_Pneumatic_System_Pressure": base_iri + "MM_1_Compressor_8",
        base_iri + "OV_1_WT_1_Pneumatic_System_Pressure": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "SM_1_Pneumatic_System_Pressure": base_iri + "SM_1_Compressor_8",
        base_iri + "VGR_1_Pneumatic_System_Pressure": base_iri + "VGR_1_Compressor_7",
    }

    for feature in feature_names:

        # Remember whether a matching iri could be found
        matched_defined_type = True
        iri_comps = []
        print("feature beginning: ", feature)
        # Derive iri from feature name
        main_component, specific_part = split(feature, '_', 2)
        iri_comps.append(main_component.upper())
        identifier, type = split(specific_part, '_', 1)

        if main_component.startswith('shop'):
            substring = split(feature, '_', 4)[1].title()
            a, b = split(substring, '_', -3)
            iri_comps = [a.upper(), b]
        elif identifier.startswith('i'):
            # Light barrier and position switches
            nbr = identifier[1]
            type = split(type, '_', 2)[0].title()
            iri_comps.append(type)
            iri_comps.append(nbr)
        elif identifier.startswith('m'):
            # Motor speeds
            nbr = identifier[1]
            iri_comps.append('Motor')
            iri_comps.append(nbr)
        elif identifier.startswith('o'):
            # Valves and compressors
            nbr = identifier[1]
            iri_comps.append(split(type, '_', 1)[0].title())
            iri_comps.append(nbr)
        elif identifier in ['current', 'target']:
            iri_comps.append('Crane_Jib')
        elif identifier == 'temper':
            # Untested because not present in dataset
            iri_comps.append('Temperature')
        elif main_component.startswith('bmx'):
            main_component = split(main_component, '_', 1)[1].upper()
            iri_comps = [main_component, identifier, 'Crane_Jib']
        elif main_component.startswith('acc'):
            # Acceleration sensors are associated with the component they observe e.g. a motor
            main_component = split(main_component, '_', 1)[1].upper()

            if type.startswith('m'):
                iri_comps = [main_component, identifier, 'Motor', type.split('_')[1]]
            elif type.startswith('comp'):
                iri_comps = [main_component, identifier, 'Compressor_8']
        elif main_component == 'sm_1' and identifier == 'detected':
            iri_comps = [main_component.upper(), 'Color', 'Sensor', '2']
        else:
            # No matching iri was found
            matched_defined_type = False

        if matched_defined_type:
            iri = base_iri + '_'.join(iri_comps)

            if iri in manual_corrections.keys():
                iri = manual_corrections.get(iri)

            feature_2_iri[feature] = iri
        else:
            # Mark the no valid iri was found for this feature
            feature_2_iri[feature] = None

        feature_2_iri[feature] = iri
        print("feature: ", feature)
    return feature_2_iri


def invert_dict(d: dict):
    """
    Creates an inverted dictionary of d: All values of a key k in d will become a key with value k in the inverted dict.
    :param d: The dictionary that should be inverted.
    :return: The resulting inverted dict.
    """

    inverted_dict = {}

    for key, value in d.items():
        if value in inverted_dict.keys():
            inverted_dict[value].append(key)
        else:
            inverted_dict[value] = [key]

    return inverted_dict


def tuple_corrections(feature_tuples, iri=None):
    # Remove self loops
    feature_tuples = [(a, b) for (a, b) in feature_tuples if a != b]

    # Add inverse, necessary because not all relations are present in the ontology
    feature_tuples.extend([(b, a) for (a, b) in feature_tuples])

    # Remove duplicates
    feature_tuples = list(set(feature_tuples))

    return feature_tuples


def store_mapping(config: Configuration, feature_2_iri: dict):
    """
    Stores the dictionary mapping features to their iri such that it can be used by other programs,
        mainly GenerateFeatureEmbeddings.py
    :param config: The configuration object.
    :param feature_2_iri: The dictionary mapping features to their iri.
    """
    print("feature_2_iri: ", feature_2_iri)
    with open(config.attribute_to_iri_mapping_file, 'w') as outfile:
        print("outfile", outfile)
        json.dump(feature_2_iri, outfile, sort_keys=True, indent=2)


def check_mapping(feature_2_iri: dict):
    """
    Ensures a iri could be determined for all features.
    :param feature_2_iri: The dictionary mapping features to their iri.
    """
    print("feature_2_iri: ", feature_2_iri)
    for feature, iri in feature_2_iri.items():
        if iri is None:
            raise ValueError(f'No IRI could be generated for feature {feature}.'
                             ' This would cause problems when trying to assign an embedding or finding relations.')


# config = configuration of file paths etc.
# feature_names: list with features (i.e. data streams)
def onto_2_matrix(config, feature_names, daemon=True, temp_id=None):
    ##############################

    # Settings which relations to include in the generated adjacency matrix.
    component_of_relation = True
    iri_relation = True
    connected_to_relation = True
    calibration_relation = True
    actuates_relation = True
    monitors_relation = True
    controls_relation = True
    isInputFor_relation = True
    sosaHosts_relation = True
    observableProperty_relation = True
    actuatesHostsProperty_relation = True
    both_precondition_same_service = False
    both_postcondition_same_service = False

    # Should not be used. Used the corresponding gsl mod instead.
    force_self_loops = False

    # Settings regarding the adj plot.
    plot_labels = True
    print_linked_features = True

    # Generate RCA features instead of ADJMat
    useFoRCA= False
    ##############################

    # Consistency check to ensure the intended configuration is used.
    if not all([component_of_relation, iri_relation, connected_to_relation, calibration_relation, actuates_relation,
                both_postcondition_same_service, both_precondition_same_service, not force_self_loops]):
        if not daemon:
            reply = input('Configuration deviates from the set standard. Continue?\n')
            if reply.lower() != 'y':
                sys.exit(-1)
        else:
            raise ValueError('Configuration deviates from the set standard.')

    if daemon and temp_id is None:
        raise ValueError('If running this as a daemon service a temporary id must be passed.')

    # importing the module
    import json

    # Opening JSON file containing a mapping from a feature name
    # (i.e. data stream or in case of RCA a component) to its IRI of the corresponding ontology
    with open(config.attribute_to_iri_mapping_file) as json_file:
        feature_2_iri = json.load(json_file)
    with open(config.component_to_iri_mapping_file) as json_file:
        component_2_iri = json.load(json_file)

    '''
    # Create dictionary that matches feature names to the matched iri
    feature_2_iri = name_2_iri(feature_names)
    print("feature_2_iri *** ", feature_2_iri)

    '''

    check_mapping(feature_2_iri)
    #store_mapping(config, feature_2_iri)

    # Invert such that iri is matched to list of associated features
    iri_2_features = invert_dict(feature_2_iri)
    iri_2_components = invert_dict(component_2_iri)

    #print("iri_2_features: ", iri_2_features)
    #print("iri_2_components: ", iri_2_components)

    # Load ontology into owlready2 for further processing (querrying, reasoning etc.)
    onto_file = config.ftonto_file
    ontology = owl.get_ontology("file://" + onto_file).load()

    # If desired, doing the reasoning first:
    # owl.JAVA_EXE = "/usr/bin/java/"
    # owl.sync_reasoner_hermit()

    #
    # The following part extracts feature - feature relations
    # or if useRCA active, then component - feature relations
    # Which relationships or pattern are considered are defined
    # at the beginning.
    #

    # Stores pairs of features in a list that are extracted based on a relationship in the ontology.
    linked_features = []
    # e.g. [('txt15_i3', 'txt15_m1.finished'), ('txt18_i3', 'txt18_m1.finished'), ... ]
    # Stores the name as string of the relationship for each pair of linked features
    responsible_relations = []
    # e.g. ['component', 'component', ... ]

    # Link features (data streams) which are matched to the same iri (entity of the ontology)
    if iri_relation:
        matched_iri_dict = {}

        for name, iri in feature_2_iri.items():
            if iri is None:
                continue

            if iri in matched_iri_dict:
                matched_iri_dict[iri].append(name)
            else:
                matched_iri_dict[iri] = [name]

        for iri, feature_lists in matched_iri_dict.items():
            feature_tuples = list(itertools.product(feature_lists, repeat=2))
            feature_tuples = tuple_corrections(feature_tuples, iri)
            linked_features.extend(feature_tuples)

            # Assign same relation for plotting
            responsible_relations.extend(['same_iri' for _ in range(len(feature_tuples))])

    print(" ---- ---- ----")
    print("Same IRI: ")
    print("responsible_relations: ", responsible_relations)
    print("linked_features: ", linked_features)
    print(" ---- ---- ----")


    if component_of_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#isComponentOf' #'FTOnto:isComponentOf'
        r2 = 'http://iot.uni-trier.de/FTOnto#hasComponent'  #'FTOnto:hasComponent'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,direct_relation=False, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1, direct_relation=False,
                                               symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['component' for _ in range(len(feature_tuples))])

    print(" ---- ---- ----")
    print("Found ", len(feature_tuples), " for hasComponent/ComponentOf pattern.")
    print("hasComponent/ComponentOf: ")
    print("feature_tuples: ", feature_tuples)
    print("responsible_relations: ", responsible_relations)
    print("linked_features: ", linked_features)
    print(" ---- ---- ----")
    print(sddssd)

    if connected_to_relation:
        r = 'http://iot.uni-trier.de/FTOnto#isConnectedTo'#'FTOnto:isConnectedTo'
        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r,
                                           direct_relation=True, symmetric_relation=True, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r,
                                               direct_relation=True, symmetric_relation=True)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['connection' for _ in range(len(feature_tuples))])

    print(" ---- ---- ----")
    print("isConnectedTo: ")
    #print("responsible_relations: ", feature_tuples)
    #print("responsible_relations: ", responsible_relations)
    #print("linked_features: ", linked_features)
    print(" ---- ---- ----")


    if calibration_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#calibrates'#'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#isCalibratedBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2)

        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['calibration' for _ in range(len(feature_tuples))])

    print(" ---- ---- ----")
    print("calibrates: ")
    #print("responsible_relations: ", feature_tuples)
    #print("responsible_relations: ", responsible_relations)
    #print("linked_features: ", linked_features)
    print(" ---- ---- ----")

    if actuates_relation:
        # Superclasses FTOnto:actuates and FTOnto:isActuatedBy not present
        '''
         actuation_relations = [
            ('FTOnto:actuatesHorizontallyForwardBackward', 'FTOnto:isActuatedHorizontallyForwardBackwardBy'),
            ('FTOnto:actuatesHorizontallyLeftRight', 'FTOnto:isActuatedHorizontallyLeftRightBy'),
            ('FTOnto:actuatesRotationallyAroundVerticalAxis', 'FTOnto:isActuatedRotationallyAroundVerticalAxisBy'),
            ('FTOnto:actuatesVertically', 'FTOnto:isActuatedVerticallyBy')
        ]
        '''
        actuation_relations = [
            ('http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy'),
            ('http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy'),
            ('http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
             'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy'),
            ('http://iot.uni-trier.de/FTOnto#actuatesVertically',
             'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy')
        ]


        for r1, r2 in actuation_relations:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=True, symmetric_relation=False, r2=r2)
            feature_tuples = tuple_corrections(feature_tuples)
            linked_features.extend(feature_tuples)

            # Assign same relation for plotting
            responsible_relations.extend(['actuation' for _ in range(len(feature_tuples))])


        print(" ---- ---- ----")
        print("actuates: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")


    if monitors_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#monitores'#'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#isMonitoredBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['monitores' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("monitors: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")


    if controls_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#controls'#'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#isControlledBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['controls' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("controls: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")


    if isInputFor_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#isInputFor'#'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#getsInputFrom'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)

        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['isInputFor' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("controls: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if sosaHosts_relation:
        r1 = 'http://www.w3.org/ns/sosa/hosts'#'FTOnto:calibrates'
        r2 = 'http://www.w3.org/ns/sosa/isHostedBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['hosts' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("hosts: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")


    if observableProperty_relation:
        r1 = 'http://www.w3.org/ns/sosa/observes'   #'FTOnto:calibrates'
        r2 = 'http://www.w3.org/ns/ssn/hasProperty' #'FTOnto:isCalibratedBy'
        r3 = 'http://www.w3.org/ns/ssn/isPropertyOf'
        r4 = 'http://www.w3.org/ns/sosa/isObservedBy'

        ''' USED FOR Quieries with a middle entity    
        SELECT ?r ?x ?r1 ?y WHERE {
        { <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?y. } 
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> . }
        }
        '''

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=False, symmetric_relation=False, r2=r2, r3=r3,r4=r4, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['obsverableProperty' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("observable property: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if actuatesHostsProperty_relation:
        actuation_relations = [
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
             'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesVertically',
             'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy',
             'http://www.w3.org/ns/sosa/hosts')
        ]

        '''
        r1 = 'http://www.w3.org/ns/sosa/isHostedBy'   #'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#actuates' #'FTOnto:isCalibratedBy'
        r3 = 'http://iot.uni-trier.de/FTOnto#isActuated'
        r4 = 'http://www.w3.org/ns/sosa/hosts'
        '''

        ''' USED FOR Quieries with a middle entity    
        SELECT ?r ?x ?r1 ?y WHERE {
        { 
        <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_3> <http://www.w3.org/ns/sosa/observes> ?x  . 
        ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?y. 
        } 
        UNION
         { 
         ?y <http://www.w3.org/ns/ssn/hasProperty> ?x. 
         ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_3> . 
         }
        }
        '''

        for r1, r2, r3, r4 in actuation_relations:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4,
                                                   iri_2_components=iri_2_components)
                print("......")
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4)
            feature_tuples = tuple_corrections(feature_tuples)
            linked_features.extend(feature_tuples)
            responsible_relations.extend(['actuate-hosts' for _ in range(len(feature_tuples))])

            print(" ---- ---- ----")
            print("actuate-hosts: ")
            # print("responsible_relations: ", feature_tuples)
            # print("responsible_relations: ", responsible_relations)
            # print("linked_features: ", linked_features)
            print(" ---- ---- ----")


    # Load the service pre- and postcondtions from a file.
    if both_precondition_same_service or both_postcondition_same_service:
        with open(config.get_additional_data_path('service_condition_pairs.json'), 'r') as f:
            service_condition_pairs = json.load(f)
            precondition_pairs = service_condition_pairs['precondition_pairs']
            postcondition_pairs = service_condition_pairs['postcondition_pairs']

    if both_precondition_same_service:
        iri_tuples = []

        for key_iri, values in precondition_pairs.items():
            iri_tuples.extend([(key_iri, value_iri) for value_iri in values])

        feature_tuples = feature_tuples_from_iri_tuples(iri_tuples, iri_2_features)
        linked_features.extend(feature_tuples)

        # Assign same relation for plotting
        responsible_relations.extend(['precondition' for _ in range(len(feature_tuples))])

    if both_postcondition_same_service:
        iri_tuples = []

        for key_iri, values in postcondition_pairs.items():
            iri_tuples.extend([(key_iri, value_iri) for value_iri in values])

        feature_tuples = feature_tuples_from_iri_tuples(iri_tuples, iri_2_features)
        linked_features.extend(feature_tuples)

        # Assign same relation for plotting
        responsible_relations.extend(['postcondition' for _ in range(len(feature_tuples))])

    if not daemon and print_linked_features:

        # Iterate of extracted feature pairs and build a data frame
        rows = []
        for a, (b, c) in zip(responsible_relations, linked_features):
            if useFoRCA:
                # Merge component and data stream mappings
                feature_2_iri_extened_by_component = feature_2_iri.copy()  # start with keys and values of x
                feature_2_iri_extened_by_component.update(component_2_iri)  # modifies z with keys and values of y
                rows.append([a, b, feature_2_iri_extened_by_component.get(b), c, feature_2_iri_extened_by_component.get(c)])
            else:
                rows.append([a, b, feature_2_iri.get(b), c, feature_2_iri.get(c)])

        df = pd.DataFrame(data=rows, columns=['Relation', 'Feature 1', 'IRI Feature 1', 'Feature 2', 'IRI Feature 2'])
        print(df.to_string())

    #
    import json
    high = ["monitores", "hosts", "actuation"]
    medium = ["component"]
    low = []
    irrelevant = []
    component_relevance = {}
    for component in component_2_iri:
        component_relevance[component] = {} #[('high',[]), ('medium',[]), ('low',[]), ('irrelevant',[])]
    for component in component_2_iri:
        #print("component: ", component)
        #compoent_relevance[component] = []
        for a, (b, c) in zip(responsible_relations, linked_features):
            #print(a,"-",b,"-",c)
            if a in high and (component == b or component ==c ):
                if (component == b):
                    component_relevance[component]['high'] = c
                else:
                    component_relevance[component]['high'] = b

            elif a in medium and (component == b or component == c ):
                if (component == b):
                    component_relevance[component]['medium'] = c
                else:
                    component_relevance[component]['medium'] = b


    print("COMPONENT:",component_relevance)

    # Generate Adj Matrix
    n = feature_names.size
    a_df = pd.DataFrame(index=feature_names, columns=feature_names, data=np.zeros(shape=(n, n)))

    for f_j, f_i in linked_features:
        if f_i != f_j:
            a_df.loc[f_i, f_j] = 1

    if force_self_loops:
        for f_i in feature_names:
            a_df.loc[f_i, f_i] = 1

    a_df.index.name = 'Features'

    if daemon:
        config.a_pre_file = f'temp/predefined_a_{temp_id}.xlsx'

        if not os.path.exists(config.get_additional_data_path('temp/')):
            os.makedirs(config.get_additional_data_path('temp/'))

        a_df.to_excel(config.get_additional_data_path(config.a_pre_file))
    else:
        a_df.to_excel(config.data_folder_prefix + 'knowledge/'+config.adj_mat_file)
        a_analysis(a_df)
        #plot(feature_names, linked_features, responsible_relations, force_self_loops, display_labels=plot_labels)
        # plot_for_thesis(feature_names, linked_features, responsible_relations)
        thesis_output(feature_2_iri, responsible_relations, linked_features)


def thesis_output(feature_2_iri, responsible_relations, linked_features):
    def shorten_feature(feature):
        limit = 35
        return feature[0:limit - 2] + '...' if len(feature) > limit else feature

    def combine_relations(relations):
        rel_2_int = {
            'no_relation': 0, 'self_loops': 1, 'component': 2, 'same_iri': 3, 'connection': 4,
            'actuation': 5, 'calibration': 6, 'precondition': 7, 'postcondition': 8,
        }

        relations = [str(rel_2_int.get(rel)) for rel in sorted(relations)]
        relations = [rel for rel in sorted(relations)]
        return ', '.join(relations)

    features, iris = [], []

    for feature, iri in feature_2_iri.items():
        features.append(feature)
        iris.append(iri)

    features = [shorten_feature(f) for f in features]
    data = np.array([features, iris]).T
    features_2_iri_df = pd.DataFrame(columns=['Datenstrom', 'IRI'], data=data)
    features_2_iri_df.index.name = 'Index'
    # features_2_iri_df = features_2_iri_df.sort_values(by='IRI', ascending=True)
    print(features_2_iri_df.to_latex(longtable=True, label='tab:streams2iri'))

    rows = []
    for a, (b, c) in zip(responsible_relations, linked_features):
        rows.append([a, b, c])

    df = pd.DataFrame(data=rows, columns=[
                      'Relation', 'Feature 1', 'Feature 2'])

    df['F1'] = df.apply(lambda x: x['Feature 1'] if x['Feature 1']
                        > x['Feature 2'] else x['Feature 2'], axis=1)
    df['F2'] = df.apply(lambda x: x['Feature 1'] if x['Feature 1']
                        < x['Feature 2'] else x['Feature 2'], axis=1)
    df = df.sort_values(by=['F1', 'F2'], ascending=False)
    df = df.drop_duplicates(subset=['F1', 'F2', 'Relation'])
    df = df.groupby(['F1', 'F2'])['Relation'].apply(
        combine_relations).reset_index()
    df = df.drop(df.loc[df['Relation'] == 'component'].index).reset_index()

    df['Datenstrom 1'] = df['F1'].apply(shorten_feature)
    df['Datenstrom 2'] = df['F2'].apply(shorten_feature)
    df = df[['Datenstrom 1', 'Relation', 'Datenstrom 2']]

    print(df.to_string())
    # print(df.to_latex(longtable=True, label='tab:relations', index=False))

def feature_tuples_from_iri_tuples(iri_tuples, iri_2_features: dict):
    feature_tuples = []

    # Create all feature pairs for each iri pair (some features are mapped to the same iri)
    for iri_1, iri_2 in iri_tuples:

        if not iri_1 in iri_2_features.keys() or not iri_2 in iri_2_features.keys():
            continue

        features_iri_1 = iri_2_features.get(iri_1)
        features_iri_2 = iri_2_features.get(iri_2)
        pairs = list(itertools.product(features_iri_1, features_iri_2))
        feature_tuples.extend(pairs)

    feature_tuples = tuple_corrections(feature_tuples)

    return feature_tuples


def a_analysis(a_df: pd.DataFrame):
    print('\nFeatures without links:')
    temp = a_df.loc[(a_df == 0).all(axis=1)]
    print(*temp.index.values, sep='\n')
    print()

    # noinspection PyArgumentList
    temp = a_df.sum(axis=0, skipna=True).sort_values(ascending=False)
    print(temp.to_string())


def prepare_query(a, r1, direct_relation, symmetric_relation, r2=None, r3=None, r4=None):
    if not direct_relation:
        assert r2 is not None, 'if not direct, a second relation must be passed'
        if r3 ==None and r4 == None:
            return "SELECT ?x WHERE {{ <" + a + "> <" + r1 + "> ?y . ?x <" + r1 + "> ?y . } " + \
                   "UNION { ?y <" + r2 + "> <" + a + "> .  ?y <" + r2 + "> ?x .}}"
        else:
            # Indirect connection such as observable property
            q = "SELECT ?y WHERE {{ <" + a + "> <" + r1 + "> ?y . ?y <" + r3 + "> ?x . } " + \
                   "UNION { ?x <" + r2 + "> ?y .  ?y <" + r4 + "> <" + a + "> .}}"
            print("q: ", q)
            return q

            ''' USED FOR Quieries with a middle entity    
            SELECT ?r ?x ?r1 ?y WHERE {
            { <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?y. } 
            UNION
             { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> . }
            }
            '''

    if symmetric_relation:
        return "SELECT ?x WHERE {{ <" + a + "> <" + r1 + "> ?x . } " + \
               " UNION { ?x <" + r1 + "> <" + a + "> . }}"
    else:
        assert r2 is not None, 'if not symmetric, a second relation must be passed'
        return "SELECT ?x WHERE {{ <" + a + "> <" + r1 + "> ?x . } " + \
               "UNION { <" + a + "> <" + r2 + "> ?x . }" + \
               "}"


def infer_connections(feature_2_iri, iri_2_features, r1, direct_relation, symmetric_relation, r2=None, r3=None, r4=None, iri_2_components=None):
    tuples = []

    iris = list(set(feature_2_iri.values()))

    for name, iri in feature_2_iri.items():
        if iri is None:
            continue

        q = prepare_query(iri, r1, direct_relation=direct_relation, symmetric_relation=symmetric_relation, r2=r2, r3=r3, r4=r4)
        #print("q: ", q)
        try:
            results = list(owl.default_world.sparql(q))
        except ValueError:
            results = []
            print(f'Query error for feature {name} with assigned IRI {iri} with query: {q}')
        #print("results: ", results)
        if not results == []:
            print("q: ", q)
            print("results: ", results)
        # Since results are in form of owl-file-name.Label, we replace them with the ontology namespace / prefix: http://iot.uni-trier.de/FTOnto#
        relevant_results = [str(res[0]).replace(config.ftonto_file_name + ".", 'http://iot.uni-trier.de/FTOnto#') for res in results]
        #relevant_results = [str(res[0]).replace('.', ':') for res in results]
        #print("relevant_results: ", relevant_results)
        relevant_results = [iri for iri in relevant_results if iri in iris]
        #print("relevant_results: ", relevant_results)
        f = []
        for res_iri in relevant_results:
            #print("res_iri: ", res_iri)
            if res_iri in iri_2_features:
                f.extend(iri_2_features.get(res_iri))
            else:
                if not iri_2_components == None:
                    if res_iri in iri_2_components:
                        f.extend(iri_2_components.get(res_iri))
                    else:
                        print("IRI not found as data stream feature: ", res_iri, " for ", name, "with relation: ", r1)

        tuples.extend([(name, res_name) for res_name in f])

    return tuples


if __name__ == '__main__':
    config = Configuration()
    dataset = Dataset(config.training_data_folder, config)
    dataset.load()

    onto_2_matrix(config, dataset.feature_names_all, daemon=False)
