"""
Varoius codependence measure: mutual info, distance correlations, variation of information
"""

from genieml_src.codependence.correlation import (angular_distance, absolute_angular_distance, squared_angular_distance, \
    distance_correlation)
from genieml_src.codependence.information import (get_mutual_info, get_optimal_number_of_bins, \
    variation_of_information_score)
from genieml_src.codependence.codependence_matrix import (get_dependence_matrix, get_distance_matrix)
from genieml_src.codependence.gnpr_distance import (spearmans_rho, gpr_distance, gnpr_distance)
from genieml_src.codependence.optimal_transport import (optimal_transport_dependence)