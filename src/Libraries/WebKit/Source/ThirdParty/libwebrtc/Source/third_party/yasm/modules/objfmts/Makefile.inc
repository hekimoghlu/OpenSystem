/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

EXTRA_DIST += modules/objfmts/dbg/Makefile.inc
EXTRA_DIST += modules/objfmts/bin/Makefile.inc
EXTRA_DIST += modules/objfmts/elf/Makefile.inc
#!EXTRA_DIST += modules/objfmts/omf/Makefile.inc
EXTRA_DIST += modules/objfmts/coff/Makefile.inc
EXTRA_DIST += modules/objfmts/macho/Makefile.inc
EXTRA_DIST += modules/objfmts/rdf/Makefile.inc
EXTRA_DIST += modules/objfmts/win32/Makefile.inc
EXTRA_DIST += modules/objfmts/win64/Makefile.inc
EXTRA_DIST += modules/objfmts/xdf/Makefile.inc

include modules/objfmts/dbg/Makefile.inc
include modules/objfmts/bin/Makefile.inc
include modules/objfmts/elf/Makefile.inc
#!include modules/objfmts/omf/Makefile.inc
include modules/objfmts/coff/Makefile.inc
include modules/objfmts/macho/Makefile.inc
include modules/objfmts/rdf/Makefile.inc
include modules/objfmts/win32/Makefile.inc
include modules/objfmts/win64/Makefile.inc
include modules/objfmts/xdf/Makefile.inc

notrans_dist_man_MANS += yasm_objfmts.7

if BUILD_MAN
yasm_objfmts.7: modules/objfmts/yasm_objfmts.xml
	$(XMLTO) -o $(top_builddir) man $(srcdir)/modules/objfmts/yasm_objfmts.xml
endif

EXTRA_DIST += modules/objfmts/yasm_objfmts.xml
