/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

libyasm_a_SOURCES += modules/objfmts/coff/coff-objfmt.c
libyasm_a_SOURCES += modules/objfmts/coff/coff-objfmt.h
libyasm_a_SOURCES += modules/objfmts/coff/win64-except.c

YASM_MODULES += objfmt_coff

$(top_srcdir)/modules/objfmts/coff/coff-objfmt.c: win64-nasm.c win64-gas.c

win64-nasm.c: $(srcdir)/modules/objfmts/coff/win64-nasm.mac genmacro$(EXEEXT)
	$(top_builddir)/genmacro$(EXEEXT) $@ win64_nasm_stdmac $(srcdir)/modules/objfmts/coff/win64-nasm.mac

BUILT_SOURCES += win64-nasm.c
CLEANFILES += win64-nasm.c
EXTRA_DIST += modules/objfmts/coff/win64-nasm.mac

win64-gas.c: $(srcdir)/modules/objfmts/coff/win64-gas.mac genmacro$(EXEEXT)
	$(top_builddir)/genmacro$(EXEEXT) $@ win64_gas_stdmac $(srcdir)/modules/objfmts/coff/win64-gas.mac

BUILT_SOURCES += win64-gas.c
CLEANFILES += win64-gas.c
EXTRA_DIST += modules/objfmts/coff/win64-gas.mac

EXTRA_DIST += modules/objfmts/coff/tests/Makefile.inc

include modules/objfmts/coff/tests/Makefile.inc
