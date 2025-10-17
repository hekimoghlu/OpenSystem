/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

# This file is include in the sub-projects's Makefiles to inherit standard
# build behavior.  The sub-project's Makefiles need to define NAME and
# VERSION.  The custompatching target may by redefined to do any patching
# just after files are unpacked from the tarball.  The postinstall target
# may be redefined to do any post install processing.

NAMEVERSION := $(NAME)-$(VERSION)
TARBALL := $(NAMEVERSION).tar.gz
DIR := $(OBJROOT)/$(NAMEVERSION)
MYSRC := $(SRCROOT)/Modules/$(NAMEVERSION)

ifeq ($(shell [ -f $(DIR)/Build.PL ] && echo YES || echo NO),YES)
PLFILE = Build.PL
MAKECMD = perl ./Build
INDIRECTPLARGS = PLBARGS
INDIRECTMAKEARGS = BUILDARGS
else
PLFILE = Makefile.PL
MAKECMD = make
INDIRECTPLARGS = PLARGS
INDIRECTMAKEARGS = MAKEARGS
endif

no_target: $(DIR)/.mark

$(DIR)/.mark: $(DIR) configure build
	touch $@

configure: custompatching
	@set -x && \
	cd $(DIR) && \
	PERL_MM_USE_DEFAULT=1 ARCHFLAGS='$(ARCHFLAGS)' $(ENV) perl $(PLFILE) $(PROJARGS) $($(INDIRECTPLARGS))

build: patchafterconfigure
	@set -x && \
	cd $(DIR) && \
	ARCHFLAGS='$(ARCHFLAGS)' $(ENV) $(MAKECMD) $($(INDIRECTMAKEARGS))

$(DIR):
	tar -xozf $(TARBALL) -C $(OBJROOT)

unpack: $(DIR)

install: realinstall postinstall

realinstall: $(DIR)/.mark
	@set -x && \
	cd $(DIR) && \
	ARCHFLAGS='$(ARCHFLAGS)' $(ENV) $(MAKECMD) install $($(INDIRECTMAKEARGS))

# allow custom patching before running Makefile.PL or Build.PL
custompatching:

# allow patching after running Makefile.PL or Build.PL
patchafterconfigure:

# allow post install customization
postinstall:

# whether to avoid 64-bit compiles
no64:
ifeq ($(NO64),YES)
	@echo YES
else
	@echo NO
endif
