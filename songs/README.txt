Adding your own song
====================
Make a .txt file here, e.g. fly_me_to_the_moon.txt, and type the kalimba tabs
from your tab sheet:

  # Song title           <- first line starting with # is the title
  1 2 3 4 5 6 7          <- middle octave: do re mi fa so la ti (C4..B4)
  1' 2' 3'               <- one dot over the number on the sheet  -> add '
  1'' 2'' 3''            <- two dots                              -> add ''
  C5 E5 G4               <- you can also write note names
  - and |                <- ignored, use them for spacing / bars

Dots written BELOW a number on some sheets mean a LOWER octave; the 17-key
C kalimba only goes down to C4 ("1"), so those notes won't exist - if your
sheet uses them it's written for a different tuning; shift the whole song up.

Run:  python air_kalimba_hands.py --song songs/fly_me_to_the_moon.txt
