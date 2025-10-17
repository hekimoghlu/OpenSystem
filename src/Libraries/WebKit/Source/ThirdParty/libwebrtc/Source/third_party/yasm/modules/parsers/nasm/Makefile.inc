/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

libyasm_a_SOURCES += modules/parsers/nasm/nasm-parser.c
libyasm_a_SOURCES += modules/parsers/nasm/nasm-parser.h
libyasm_a_SOURCES += modules/parsers/nasm/nasm-parser-struct.h
libyasm_a_SOURCES += modules/parsers/nasm/nasm-parse.c
nodist_libyasm_a_SOURCES += nasm-token.c

YASM_MODULES += parser_nasm parser_tasm

nasm-token.c: $(srcdir)/modules/parsers/nasm/nasm-token.re re2c$(EXEEXT)
	$(top_builddir)/re2c$(EXEEXT) -b -o $@ $(srcdir)/modules/parsers/nasm/nasm-token.re

BUILT_SOURCES += nasm-token.c

CLEANFILES += nasm-token.c

EXTRA_DIST += modules/parsers/nasm/nasm-token.re

$(top_srcdir)/modules/parsers/nasm/nasm-parser.c: nasm-macros.c

nasm-macros.c: $(srcdir)/modules/parsers/nasm/nasm-std.mac genmacro$(EXEEXT)
	$(top_builddir)/genmacro$(EXEEXT) $@ nasm_standard_mac $(srcdir)/modules/parsers/nasm/nasm-std.mac

BUILT_SOURCES += nasm-macros.c
CLEANFILES += nasm-macros.c
EXTRA_DIST += modules/parsers/nasm/nasm-std.mac

EXTRA_DIST += modules/parsers/nasm/tests/Makefile.inc

include modules/parsers/nasm/tests/Makefile.inc
