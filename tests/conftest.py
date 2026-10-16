from pathlib import Path

import pytest


SPEC_FIXTURE = """\
#F sample.spec
#E 1700000000
#D Mon Jan 01 00:00:00 2024
#C user = isaac

#O0 mot_a mot_b mot_c
#O1 mot_d mot_e

#S 1 ascan mot_a 0 1 10 1
#D Mon Jan 01 12:34:56 2024
#P0 0.1 0.2 0.3
#P1 0.4 0.5
#UXML <ad_detector><sample_detector_distance>123.45</sample_detector_distance></ad_detector>
#UXML <ad_center><beam_center_x>512</beam_center_x><beam_center_y>256</beam_center_y></ad_center>
#UXML <ad_regions_of_interest><roi><name>roi1</name><x>10</x><min_x>500</min_x><size_x>24</size_x><min_y>250</min_y><size_y>12</size_y></roi><roi><name>roi2</name><x>20</x></roi></ad_regions_of_interest>
#N 3
#L col_a col_b col_c
0.0 1.0 2.0
0.5 1.5 2.5
1.0 2.0 3.0

#S 2 ascan mot_b -1 1 5 0.5
#D Tue Feb 02 01:02:03 2024
#P0 1.1 1.2 1.3
#P1 1.4 1.5
#L only_one_col
10.0
20.0
"""


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.spec"
    path.write_text(SPEC_FIXTURE, encoding="utf-8")
    return path
