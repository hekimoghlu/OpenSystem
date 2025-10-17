/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

libyasm_a_SOURCES += modules/parsers/gas/gas-parser.c
libyasm_a_SOURCES += modules/parsers/gas/gas-parser.h
libyasm_a_SOURCES += modules/parsers/gas/gas-parse.c
libyasm_a_SOURCES += modules/parsers/gas/gas-parse-intel.c
nodist_libyasm_a_SOURCES += gas-token.c

YASM_MODULES += parser_gas parser_gnu

gas-token.c: $(srcdir)/modules/parsers/gas/gas-token.re re2c$(EXEEXT)
	$(top_builddir)/re2c$(EXEEXT) -b -o $@ $(srcdir)/modules/parsers/gas/gas-token.re

BUILT_SOURCES += gas-token.c

CLEANFILES += gas-token.c

EXTRA_DIST += modules/parsers/gas/tests/Makefile.inc
EXTRA_DIST += modules/parsers/gas/gas-token.re

include modules/parsers/gas/tests/Makefile.inc
