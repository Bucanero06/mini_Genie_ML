"""
Varoius codependence measure: mutual info, distance correlations, variation of information
"""

from Modules.codependence_algorythms.correlation import (angular_distance, absolute_angular_distance, squared_angular_distance, \
                                                         distance_correlation)
from Modules.codependence_algorythms.information import (get_mutual_info, get_optimal_number_of_bins, \
                                                         variation_of_information_score)
from Modules.codependence_algorythms.codependence_matrix import (get_dependence_matrix, get_distance_matrix)
from Modules.codependence_algorythms.gnpr_distance import (spearmans_rho, gpr_distance, gnpr_distance)
from Modules.codependence_algorythms.optimal_transport import (optimal_transport_dependence)