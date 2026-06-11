.PHONY: simulate interp compile interactive layout assemble test dfg preprocess_dfg
LUCID_DIR = ~/Desktop/lucid
OPT_DIR = ..

# trace optimize w/ constant symbolics
simulate:
	python3 $(OPT_DIR)/simulator.py rotating_cmsopt.json

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state --suppress-event-reports rotating_cms.dpt

# run interpreter in interactive mode
interactive:
	$(LUCID_DIR)/dpt -i --suppress-final-state --suppress-event-reports rotating_cms.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc rotating_cms.dpt

#layout
layout:
	$(LUCID_DIR)/bin/layout.py rotating_cms.dpt
#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py rotating_cms.dpt  --dfg


# compile P4 to tofino
assemble:
	cd lucid_tofino_build; make lucid_tofino_build	 

# test P4 on the asic model
test:
	cd lucid_tofino_build; make test ../rotating_cms.json
