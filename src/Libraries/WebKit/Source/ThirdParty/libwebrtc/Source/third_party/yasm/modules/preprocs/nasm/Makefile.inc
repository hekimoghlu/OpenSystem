/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

libyasm_a_SOURCES += modules/preprocs/nasm/nasm-preproc.c
libyasm_a_SOURCES += modules/preprocs/nasm/nasm-pp.h
libyasm_a_SOURCES += modules/preprocs/nasm/nasm-pp.c
libyasm_a_SOURCES += modules/preprocs/nasm/nasm.h
libyasm_a_SOURCES += modules/preprocs/nasm/nasmlib.h
libyasm_a_SOURCES += modules/preprocs/nasm/nasmlib.c
libyasm_a_SOURCES += modules/preprocs/nasm/nasm-eval.h
libyasm_a_SOURCES += modules/preprocs/nasm/nasm-eval.c

YASM_MODULES += preproc_nasm preproc_tasm

$(top_srcdir)/modules/preprocs/nasm/nasm-preproc.c: nasm-version.c

nasm-version.c: version.mac genmacro$(EXEEXT)
	$(top_builddir)/genmacro$(EXEEXT) $@ nasm_version_mac version.mac

BUILT_SOURCES += nasm-version.c
CLEANFILES += nasm-version.c

version.mac: genversion$(EXEEXT)
	$(top_builddir)/genversion$(EXEEXT) $@

BUILT_SOURCES += version.mac
CLEANFILES += version.mac

noinst_PROGRAMS += genversion

genversion_SOURCES =
EXTRA_DIST += modules/preprocs/nasm/genversion.c
genversion_LDADD = genversion.$(OBJEXT)
genversion_LINK = $(CCLD_FOR_BUILD) -o $@

genversion.$(OBJEXT): modules/preprocs/nasm/genversion.c
	$(CC_FOR_BUILD) $(DEFAULT_INCLUDES) $(INCLUDES) -c -o $@ `test -f modules/preprocs/nasm/genversion.c || echo '$(srcdir)/'`modules/preprocs/nasm/genversion.c

EXTRA_DIST += modules/preprocs/nasm/tests/Makefile.inc

include modules/preprocs/nasm/tests/Makefile.inc
