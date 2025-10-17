/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
# just before the modules are built.  The postinstall target
# may be redefined to do any post install processing.

NAMEVERSION := $(NAME)-$(VERSION)
DIR := $(OBJROOT)/$(NAMEVERSION)

INCLUDEPYTHONVERS = $(PYPREFIX)/Headers
export PYFRAMEWORKNAME = $(shell echo $(PYPREFIX) | cut -d '/' -f5)
export PYAPPNAME = $(shell echo '$(PYFRAMEWORKNAME)' | sed 's,\.framework,,')
export PYEXECUTABLE = $(PYPREFIX)/Resources/Python.app/Contents/MacOS/$(PYAPPNAME)

ifeq ($(TOPDIR),AppleInternal)
export LIBRARYPREFIX = /AppleInternal/Library
export __INTERNALSDKBUILD__ = YES
else
export LIBRARYPREFIX = /Library
endif

$(DIR):
	cp -a $(NAMEVERSION) $(OBJROOT)

install: realinstall postinstall

realinstall: $(DIR) patchall custompatching
	cd $(DIR) && \
	ARCHFLAGS="$(RC_CFLAGS) -iwithsysroot /usr/local/libressl-3.3/include -L$(SDKROOT)/usr/local/libressl-3.3/lib" \
	python setup.py install --home="$(EXTRAS)" --root="$(DSTROOT)" $(SETUPARGS)

# allow custom patching before running setup.py
custompatching:

# install custom patch on all modules before running setup.py
patchall:
	sed -e 's,@SDKROOT@,$(SDKROOT),g' -e 's,@INCLUDEPYTHONVERS@,$(INCLUDEPYTHONVERS),g' $(SRCROOT)/Modules/setup.cfg.patch >> $(DIR)/setup.cfg

# allow post install customization
postinstall:

# whether to avoid 64-bit compiles
no64:
ifeq ($(NO64),YES)
	@echo YES
else
	@echo NO
endif
