#!/bin/bash

doxygen docs/Doxyfile
cd docs/latex
make
cp refman.pdf "../../Potential Field Avoidance Documentation.pdf"
