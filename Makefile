# Executables
YARN = yarn
VG2PNG = ./node_modules/vega-cli/bin/vg2png
VL2VG = ./node_modules/vega-lite/bin/vl2vg

# Options
YARNFLAGS = --no-progress --frozen-lockfile

# Plot files
vl-json := $(wildcard out/*.vl.json)
pngs := $(vl-json:.vl.json=.png)

all: plots

info:
	$(info $$vg2png is [${vg2png}])

clean:
	rm -f $(pngs)

plots: $(vg2png) $(vl2vg) $(pngs)

$(vg2png):
	$(YARN) install $(YARNFLAGS)

$(vl2vg):
	$(YARN) install $(YARNFLAGS)

%.vg.json: $(vl2vg) %.vl.json
	$(VL2VG) $< $@

%.png: $(vg2png) %.vg.json
	$(VG2PNG) $< $@
