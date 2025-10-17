/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

TESTS += modules/objfmts/win64/tests/win64_test.sh

EXTRA_DIST += modules/objfmts/win64/tests/win64_test.sh
EXTRA_DIST += modules/objfmts/win64/tests/sce1.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce1.hex
EXTRA_DIST += modules/objfmts/win64/tests/sce1-err.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce1-err.errwarn
EXTRA_DIST += modules/objfmts/win64/tests/sce2.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce2.hex
EXTRA_DIST += modules/objfmts/win64/tests/sce2-err.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce2-err.errwarn
EXTRA_DIST += modules/objfmts/win64/tests/sce3.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce3.hex
EXTRA_DIST += modules/objfmts/win64/tests/sce3.masm
EXTRA_DIST += modules/objfmts/win64/tests/sce4.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce4.hex
EXTRA_DIST += modules/objfmts/win64/tests/sce4.masm
EXTRA_DIST += modules/objfmts/win64/tests/sce4-err.asm
EXTRA_DIST += modules/objfmts/win64/tests/sce4-err.errwarn
EXTRA_DIST += modules/objfmts/win64/tests/win64-abs.asm
EXTRA_DIST += modules/objfmts/win64/tests/win64-abs.hex
EXTRA_DIST += modules/objfmts/win64/tests/win64-curpos.asm
EXTRA_DIST += modules/objfmts/win64/tests/win64-curpos.hex
EXTRA_DIST += modules/objfmts/win64/tests/win64-dataref.asm
EXTRA_DIST += modules/objfmts/win64/tests/win64-dataref.hex
EXTRA_DIST += modules/objfmts/win64/tests/win64-dataref.masm
EXTRA_DIST += modules/objfmts/win64/tests/win64-dataref2.asm
EXTRA_DIST += modules/objfmts/win64/tests/win64-dataref2.hex
EXTRA_DIST += modules/objfmts/win64/tests/win64-dataref2.masm
EXTRA_DIST += modules/objfmts/win64/tests/win64-function.asm
EXTRA_DIST += modules/objfmts/win64/tests/win64-function.hex
EXTRA_DIST += modules/objfmts/win64/tests/win64-imagebase.hex
EXTRA_DIST += modules/objfmts/win64/tests/win64-imagebase.asm

EXTRA_DIST += modules/objfmts/win64/tests/gas/Makefile.inc

include modules/objfmts/win64/tests/gas/Makefile.inc
