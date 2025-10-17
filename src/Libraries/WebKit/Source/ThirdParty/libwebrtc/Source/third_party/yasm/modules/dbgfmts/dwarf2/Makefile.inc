/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

libyasm_a_SOURCES += modules/dbgfmts/dwarf2/dwarf2-dbgfmt.h
libyasm_a_SOURCES += modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c
libyasm_a_SOURCES += modules/dbgfmts/dwarf2/dwarf2-line.c
libyasm_a_SOURCES += modules/dbgfmts/dwarf2/dwarf2-aranges.c
libyasm_a_SOURCES += modules/dbgfmts/dwarf2/dwarf2-info.c

YASM_MODULES += dbgfmt_dwarf2

EXTRA_DIST += modules/dbgfmts/dwarf2/tests/Makefile.inc
include modules/dbgfmts/dwarf2/tests/Makefile.inc
