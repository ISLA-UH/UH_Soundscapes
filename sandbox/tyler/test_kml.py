"""
Test the kml_files functions
"""

from sandbox.tyler.kml_files import write_kml, load_kml

data_dict = {'testandroid1': {'mean acc': 5.0, 'mean alt': 11.800000000000002, 'mean bar': 101.605, 
                              'mean lat': 19.72843, 'mean lon': -156.05909999999997, 'os': 'Android', 
                              'sample rate': 800.0, 'std acc': 0.0, 'std alt': 0.09999999999999964, 
                              'std bar': 0.004082482904640718, 'std lat': 9.999999999976694e-05, 
                              'std lon': 0.00010000000000331966}, 
             'testios1': {'mean acc': 5.0, 'mean alt': 11.800000000000002, 'mean bar': 101.605, 
                          'mean lat': 19.72843, 'mean lon': -156.05909999999997, 'os': 'iOS', 
                          'sample rate': 800.0, 'std acc': 0.0, 'std alt': 0.09999999999999964, 
                          'std bar': 0.004082482904640718, 'std lat': 9.999999999976694e-05, 
                          'std lon': 0.00010000000000331966}
            }

current_kml_path = 'sandbox/tyler/test.kml'

write_kml(current_kml_path, data_dict)

loaded = load_kml(current_kml_path)

print(loaded)

exit(1)