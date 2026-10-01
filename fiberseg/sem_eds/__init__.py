"""SEM tiling -> fibre segmentation -> targeted EDS on a Bruker ESPRIT/QUANTAX system.

Integrated from the sem_eds_handoff skeleton. Phases:
  0  bruker_api.py, microscope.py, settings.py - ctypes bindings + backends
  1  segmenter.py                               - fiberseg checkpoint as segmenter
  2  tests/test_sem_eds.py                      - offline tests
  3  bringup/step1..step6                       - instrument bring-up, run in order
  4  run_tiles.py                               - the closed loop
  5  classify.py                                - spectrum classification
"""
