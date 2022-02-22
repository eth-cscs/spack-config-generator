PATH := $(CURDIR)/eth-cscs-spack/bin:$(PATH)

# Disable ~/.spack and /etc/spack config, just to be sure.
# and make sure the current directory is used for any spack
# caches, if that's somehow triggered.
SPACK_DISABLE_LOCAL_CONFIG:=1
SPACK_USER_CACHE_DIR:=$(CURDIR)/.spack
CONFIG_FLAGS:=

.PHONY: check-spack install clean

all: generated-configs

eth-cscs-spack.tar.gz:
	curl -Lfso $@ https://github.com/eth-cscs/spack/archive/refs/heads/develop.tar.gz

eth-cscs-spack: eth-cscs-spack.tar.gz
	mkdir $@ && tar -xf $< --strip-components=1 -C $@

check-spack: eth-cscs-spack
	@echo using $$(which spack) at $$(spack --version)

generated-configs: eth-cscs-spack | check-spack 
	./spack-allinone.py $(CONFIG_FLAGS)

install: generated-configs
	install -Dm 644 spack-config "$(DESTDIR)/modules/spack-config/1.0.0"
	cd $< && find . -type f -exec install -Dm 644 '{}' "$(DESTDIR)/{}" \;

clean:
	rm -rf .spack generated-configs eth-cscs-spack*

