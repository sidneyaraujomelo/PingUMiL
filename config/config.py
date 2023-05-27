import os
import errno
import json
import random
import os

'''if (len(attrib_name_list) != len(attrib_type_list)):
    print("Attributes name and type list are not the same size.")
    print(len(attrib_name_list))
    print(len(attrib_type_list))
    print([x for x in zip(attrib_name_list,attrib_type_list)])

if (len(attrib_name_list) != len(attrib_default_value_list)):
    print("Attributes name and type list are not the same size.")
    print(len(attrib_name_list))
    print(len(attrib_default_value_list))
    print([x for x in zip(attrib_name_list,attrib_default_value_list)])'''

def find_node_with_tag(element, tag):
    #print(element)
    if element.tag == tag:
        return element
    else:
        #print(element)
        for child in element:
            found_node = find_node_with_tag(child, tag)
            if found_node is not None:
                return found_node
    return None


def find_attribute_node_with_name(element, name):
    for attribute in element.iter("attribute"):
        if attribute.find('name').text == name:
            return attribute
    return None

def prepareDictionaries():
    # Dictionary of type Classes->{Class1, Class2, Class3}
    cat_attrib_list = []
    for i, atb_type in enumerate(attrib_type_list):
        if atb_type != 'categoric':
            continue
        sup_dict[attrib_name_list[i]] = []
        cat_attrib_list.append(attrib_name_list[i])
    for i, tag_type in enumerate(tag_type_list):
        if tag_type != 'categoric':
            continue
        sup_dict[tag_name_list[i]] = []
        cat_attrib_list.append(tag_name_list[i])
    return cat_attrib_list


def populateCategoricDictionaries(root,cat_attrib_list):
    # Traverse the tree until find vertices node
    vertices = find_node_with_tag(root, "vertices")
    # Iterates over vertex nodes
    for vertex in vertices.iter("vertex"):
        # Get tags of the vertex
        for cat_attrib in cat_attrib_list:
            # Find tag value for a given attribute name
            if (vertex.find(cat_attrib) != None):
                # If value exists
                tag_value = vertex.find(cat_attrib).text
                if (tag_value == None):
                    tag_value = tag_default_value_list[tag_name_list.index(cat_attrib)]
                if (tag_value not in sup_dict[cat_attrib]):
                    sup_dict[cat_attrib].append(tag_value)
        # Get attributes node
        attributes = vertex.find("attributes")
        # iterates over the list of wanted attributes
        for cat_attrib in cat_attrib_list:
            #Skip if it tries to search for a provenance tag
            if cat_attrib in tag_name_list:
                continue
            # Find attribute node for a given attribute name
            attrib_node = find_attribute_node_with_name(attributes, cat_attrib)
            if (attrib_node == None):
                current_node_atb_value = attrib_default_value_list[attrib_name_list.index(cat_attrib)]
            else:
                # Get the value of the attribute node for a given attribute name
                current_node_atb_value = attrib_node.find('value').text
            # Adds the value to the dictionary if it's not there yet
            if (current_node_atb_value not in sup_dict[cat_attrib]):
                sup_dict[cat_attrib].append(current_node_atb_value)


def buildOneHotVectorRepresentationForCategoricAttributes(cat_attrib_list):
    # Now we have all categories for every categoric attribute in the sup_dict
    # Iterate again over all categoric attribute
    for cat_attrib in cat_attrib_list:
        # Create a dictionary for every attribute
        category_representations = {}
        # Iterate over every possible value of the attribute, generating its 1-hot-vector representation
        for idx, category in enumerate(sup_dict[cat_attrib]):
            category_representations[category] = [0] * len(sup_dict[cat_attrib])
            category_representations[category][idx] = 1
        # Categoric_att_dict now points to every categoric attribute entry
        # And each Categoric Attribute Entry points to its 1-hot-vector representation
        categoric_att_dict[cat_attrib] = category_representations


#to be deprecated
def build_categoric_dictionary(root):
    print("Building Dictionary of Categoric Attributes and One-Hot-Vector representations")
    # Dictionary of type Classes->{Class1, Class2, Class3}
    cat_attrib_list = prepareDictionaries()

    populateCategoricDictionaries(root, cat_attrib_list)

    buildOneHotVectorRepresentationForCategoricAttributes(cat_attrib_list)

    print("One-hot-vector representations of every categoric attribute: ")
    print(categoric_att_dict)

#to be deprecated
def build_categoric_dictionary_for_list(xmls):
    print("Building Dictionary of Categoric Attributes and One-Hot-Vector representations")
    cat_attrib_list = prepareDictionaries()

    # iterate over xml files
    for root in xmls:
        populateCategoricDictionaries(root, cat_attrib_list)

    buildOneHotVectorRepresentationForCategoricAttributes(cat_attrib_list)

    print("One-hot-vector representations of every categoric attribute: ")
    print(categoric_att_dict)


def get_ohv_for_attribute(categoric_att_dict, attribute_name, attribute_value):
    #print(attribute_name)
    return categoric_att_dict[attribute_name][attribute_value]


def dictToNpArrayofArrays(featsmap):
    arr = np.array([v for k, v in featsmap.items()])
    print(arr)
    return arr

