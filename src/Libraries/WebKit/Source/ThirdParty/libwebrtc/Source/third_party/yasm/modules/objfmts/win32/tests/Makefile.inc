/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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

TESTS += modules/objfmts/win32/tests/win32_test.sh

EXTRA_DIST += modules/objfmts/win32/tests/export.asm
EXTRA_DIST += modules/objfmts/win32/tests/export.hex
EXTRA_DIST += modules/objfmts/win32/tests/win32_test.sh
EXTRA_DIST += modules/objfmts/win32/tests/win32-curpos.asm
EXTRA_DIST += modules/objfmts/win32/tests/win32-curpos.hex
EXTRA_DIST += modules/objfmts/win32/tests/win32-overdef.asm
EXTRA_DIST += modules/objfmts/win32/tests/win32-overdef.hex
EXTRA_DIST += modules/objfmts/win32/tests/win32-safeseh.asm
EXTRA_DIST += modules/objfmts/win32/tests/win32-safeseh.hex
EXTRA_DIST += modules/objfmts/win32/tests/win32-safeseh.masm
EXTRA_DIST += modules/objfmts/win32/tests/win32-segof.asm
EXTRA_DIST += modules/objfmts/win32/tests/win32-segof.hex
EXTRA_DIST += modules/objfmts/win32/tests/win32test.c
EXTRA_DIST += modules/objfmts/win32/tests/win32test.asm
EXTRA_DIST += modules/objfmts/win32/tests/win32test.hex

EXTRA_DIST += modules/objfmts/win32/tests/gas/Makefile.inc

include modules/objfmts/win32/tests/gas/Makefile.inc
