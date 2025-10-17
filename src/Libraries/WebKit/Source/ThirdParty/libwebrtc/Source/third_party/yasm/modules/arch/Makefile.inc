/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

EXTRA_DIST += modules/arch/x86/Makefile.inc
EXTRA_DIST += modules/arch/lc3b/Makefile.inc

include modules/arch/x86/Makefile.inc
include modules/arch/lc3b/Makefile.inc

notrans_dist_man_MANS += yasm_arch.7

if BUILD_MAN
yasm_arch.7: modules/arch/yasm_arch.xml
	$(XMLTO) -o $(top_builddir) man $(srcdir)/modules/arch/yasm_arch.xml
endif

EXTRA_DIST += modules/arch/yasm_arch.xml
