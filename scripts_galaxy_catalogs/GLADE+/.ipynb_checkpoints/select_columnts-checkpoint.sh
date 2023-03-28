#!/bin/bash
# C1 Hyperleda flag, C2 Wise flag, C3 Quasar and cluster flag, C4 RA, C5 Dec, C6 m_B, C7 m_K, C8 redshift, C9 Peculiar velocity correction, C10 redshift and lum distance flag 
# awk '{print $3,$6,$8,$9,$10,$11,$18,$24,$25,$28}' GLADE+.txt > GLADE+_reduced.txt
#awk 'NR==1, NR==10 {print $4,$6,$8,$9,$10,$11,$18,$20,$25,$26,$27,$30}' GLADE+.txt > GLADE+_reduced.txt

# Do not look above this line
#-----------------------------------------------------------------------------------------------------------

# Columns extracted from the column description of GLADE.
# C1 Hyperleda flag, C2 Wise flag, C3 Quasar and cluster flag, C4 RA, C5 Dec, C6 m_B, C7 m_K,C8 m_W1, C9 redshift, C10 Peculiar velocity correction,C11 error peculiar velocity,C12 redshift and lum distance flag 
awk '{print $4,$6,$8,$9,$10,$11,$18,$20,$25,$26,$27,$30}' GLADE+.txt > GLADE+_reduced.txt
