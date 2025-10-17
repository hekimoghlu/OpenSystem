/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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

EXTRA_DIST += modules/dbgfmts/codeview/Makefile.inc
EXTRA_DIST += modules/dbgfmts/dwarf2/Makefile.inc
EXTRA_DIST += modules/dbgfmts/null/Makefile.inc
EXTRA_DIST += modules/dbgfmts/stabs/Makefile.inc

include modules/dbgfmts/codeview/Makefile.inc
include modules/dbgfmts/dwarf2/Makefile.inc
include modules/dbgfmts/null/Makefile.inc
include modules/dbgfmts/stabs/Makefile.inc

notrans_dist_man_MANS += yasm_dbgfmts.7

if BUILD_MAN
yasm_dbgfmts.7: modules/dbgfmts/yasm_dbgfmts.xml
	$(XMLTO) -o $(top_builddir) man $(srcdir)/modules/dbgfmts/yasm_dbgfmts.xml
endif

EXTRA_DIST += modules/dbgfmts/yasm_dbgfmts.xml
