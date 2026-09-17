# Time-layered predictive risk field, published by risk_perception's
# predictive_risk_costmap_node and consumed by panoptex_nav's PredictedRiskCritic.
#
# Layer k covers the instant  header.stamp + horizon_start + k * dt  (seconds).
# Layer 0 is "now" (horizon_start is normally 0.0).  Values are 0-100 risk as in
# nav_msgs/OccupancyGrid (-1 = unknown).  Cell (row r, col c) of layer k lives at
# data[k * info.height * info.width + r * info.width + c], i.e. the same row-major
# convention as OccupancyGrid.data, so each layer can be sliced straight into one.
#
# The field is EGO-INDEPENDENT on purpose: no closest-point-of-approach factor is
# applied (unlike /risk_costmap_predictive).  A controller critic that scores a
# candidate trajectory pose at its own time offset supplies the ego motion itself.

std_msgs/Header header          # frame_id = the risk grid frame (map)
nav_msgs/MapMetaData info       # resolution / width / height / origin, all layers
float32 dt                      # seconds between consecutive layers
uint8 steps                     # number of layers (>= 1)
float32 horizon_start           # seconds from header.stamp to layer 0 (usually 0.0)
int8[] data                     # steps * info.height * info.width
