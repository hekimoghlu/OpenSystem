/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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

libyasm_a_SOURCES += modules/arch/lc3b/lc3barch.c
libyasm_a_SOURCES += modules/arch/lc3b/lc3barch.h
libyasm_a_SOURCES += modules/arch/lc3b/lc3bbc.c
nodist_libyasm_a_SOURCES += lc3bid.c

YASM_MODULES += arch_lc3b

lc3bid.c: $(srcdir)/modules/arch/lc3b/lc3bid.re re2c$(EXEEXT)
	$(top_builddir)/re2c$(EXEEXT) -s -o $@ $(srcdir)/modules/arch/lc3b/lc3bid.re

CLEANFILES += lc3bid.c

EXTRA_DIST += modules/arch/lc3b/tests/Makefile.inc
EXTRA_DIST += modules/arch/lc3b/lc3bid.re

include modules/arch/lc3b/tests/Makefile.inc
