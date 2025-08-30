"""
KML related stuff goes here
TODO: fix imports and requirements
"""
import pathlib
from typing import Dict

from fastkml import kml, styles
from fastkml.features import Placemark
from fastkml.utils import find_all
from pygeoif.geometry import Point as pyPoint


def load_kml(kml_file: str) -> Dict[str, Dict[str, float]]:
    """
    load location from a kml file
    :param kml_file: full path of the file to load data from
    :return: dictionary of locations with identifiers
    """
    kml_data = kml.KML.parse(kml_file)
    locations: list[Placemark] = list(find_all(kml_data, of_type=Placemark))
    set_locations = {}
    for place in locations:
        set_locations[place.name] = {"lon": place.geometry.x, "lat": place.geometry.y, "alt": place.geometry.z}
    return set_locations


def write_kml(kml_file: str, master_dict: Dict[str, Dict[str, float]]):
    """
    put information from master_dict into a kml file
    :param kml_file: full path of kml file to write data to
    :param master_dict: the dictionary of information to write
    """
    ns = "{http://www.opengis.net/kml/2.2}"
    # declare kml structure and the document
    kmlz = kml.KML(ns=ns)
    # declare, then add styles to doc
    pnt_style = styles.IconStyle(id="is3", color="ff0000ff",
                                 icon_href="http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")
    doc_style = styles.Style(id="s2", styles=[pnt_style])
    doc = kml.Document(ns, id="d1", styles=[doc_style])
    # id is assigned dynamically as new elements are created
    new_id = 4
    for key in master_dict.keys():
        # todo: how do we know if bar is better than alt?
        # set point description to os and sample rate
        description = "{} {}hz".format(master_dict[key]["os"], str(master_dict[key]["sample rate"]))
        # declare the placemark, then give it some coordinates
        new_point = pyPoint(x=master_dict[key]["mean lon"],
                            y=master_dict[key]["mean lat"],
                            z=master_dict[key]["mean alt"])
        pnt = kml.Placemark(ns, id=str(f"p{new_id}"), name=key, description=description,
                            style_url=styles.StyleUrl(url="#s2"), geometry=new_point)
        new_id += 1
        # add placemark to doc
        doc.append(pnt)
    # add the doc to the kml file
    kmlz.append(doc)
    # write the kml file, with nice formatting
    kmlz.write(pathlib.Path(kml_file), precision=5, prettyprint=True)