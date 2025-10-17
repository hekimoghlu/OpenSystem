/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

bin_PROGRAMS += yasm
dist_man_MANS += yasm.1

if BUILD_MAN
yasm.1: frontends/yasm/yasm.xml
	$(XMLTO) -o $(top_builddir) man $(srcdir)/frontends/yasm/yasm.xml
endif

yasm_SOURCES  = frontends/yasm/yasm.c
yasm_SOURCES += frontends/yasm/yasm-options.c
yasm_SOURCES += frontends/yasm/yasm-options.h

$(srcdir)/frontends/yasm/yasm.c: license.c

license.c: $(srcdir)/COPYING genstring$(EXEEXT)
	$(top_builddir)/genstring$(EXEEXT) license_msg $@ $(srcdir)/COPYING

BUILT_SOURCES += license.c
CLEANFILES += license.c

yasm_LDADD = libyasm.a $(INTLLIBS)

EXTRA_DIST += frontends/yasm/yasm.xml
