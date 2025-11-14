######################################################################
#
# DESCRIPTION: Verilator Example: Small Makefile
#
# This calls the object directory makefile.  That allows the objects to
# be placed in the "current directory" which simplifies the Makefile.
#
# This file ONLY is placed under the Creative Commons Public Domain, for
# any use, without warranty, 2020 by Wilson Snyder.
# SPDX-License-Identifier: CC0-1.0
#
######################################################################
# Check for sanity to avoid later confusion

ifneq ($(words $(CURDIR)),1)
 $(error Unsupported: GNU Make cannot build in directories containing spaces, build elsewhere: '$(CURDIR)')
endif

######################################################################
# Set up variables

# If $VERILATOR_ROOT isn't in the environment, we assume it is part of a
# package install, and verilator is in your path. Otherwise find the
# binary relative to $VERILATOR_ROOT (such as when inside the git sources).
ifeq ($(VERILATOR_ROOT),)
VERILATOR = verilator
VERILATOR_COVERAGE = verilator_coverage
else
export VERILATOR_ROOT
VERILATOR = $(VERILATOR_ROOT)/bin/verilator
VERILATOR_COVERAGE = $(VERILATOR_ROOT)/bin/verilator_coverage
endif

# Generate C++ in executable form
VERILATOR_FLAGS = -cc --exe -x-assign fast -Wno-fatal --assert --coverage-line \
                 -Wno-WIDTHEXPAND -Wno-BLKSEQ -Wno-VARHIDDEN \
                 -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-CASEOVERLAP \
                 -Wno-BLKANDNBLK -Wno-COMBDLY -Wno-PROCASSWIRE \
                 -Wno-ASCRANGE -Wno-LATCH -Wno-UNDRIVEN -Wno-UNUSEDGENVAR \
                 -Wno-PINMISSING -Wno-IMPLICIT -Wno-DECLFILENAME \
                 -Wno-UNSIGNED -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT \
                 -Wno-MULTIDRIVEN --no-std \
                 -CFLAGS -faligned-new -CFLAGS -Wno-sign-compare -CFLAGS -Wno-overflow

# Input files for Verilator
VERILATOR_INPUT = -f input.vc top_module.v sim-main.cpp rfuzz-harness.cpp

######################################################################
default: run

run:
	$(VERILATOR) $(VERILATOR_FLAGS) $(VERILATOR_INPUT)
	cd obj_dir && make -f Vtop_module.mk
	obj_dir/Vtop_module

clean:
	rm -rf obj_dir logs *.log *.dmp *.vpd coverage.dat core
	rm -f *.o

show-config:
	$(VERILATOR) -V
