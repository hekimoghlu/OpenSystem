/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

TESTS += modules/objfmts/bin/tests/bin_test.sh

EXTRA_DIST += modules/objfmts/bin/tests/bin_test.sh
EXTRA_DIST += modules/objfmts/bin/tests/abs.asm
EXTRA_DIST += modules/objfmts/bin/tests/abs.hex
EXTRA_DIST += modules/objfmts/bin/tests/bigorg.asm
EXTRA_DIST += modules/objfmts/bin/tests/bigorg.hex
EXTRA_DIST += modules/objfmts/bin/tests/bigorg.errwarn
EXTRA_DIST += modules/objfmts/bin/tests/bin-farabs.asm
EXTRA_DIST += modules/objfmts/bin/tests/bin-farabs.hex
EXTRA_DIST += modules/objfmts/bin/tests/bin-rip.asm
EXTRA_DIST += modules/objfmts/bin/tests/bin-rip.hex
EXTRA_DIST += modules/objfmts/bin/tests/bintest.asm
EXTRA_DIST += modules/objfmts/bin/tests/bintest.hex
EXTRA_DIST += modules/objfmts/bin/tests/float-err.asm
EXTRA_DIST += modules/objfmts/bin/tests/float-err.errwarn
EXTRA_DIST += modules/objfmts/bin/tests/float.asm
EXTRA_DIST += modules/objfmts/bin/tests/float.hex
EXTRA_DIST += modules/objfmts/bin/tests/integer-warn.asm
EXTRA_DIST += modules/objfmts/bin/tests/integer-warn.hex
EXTRA_DIST += modules/objfmts/bin/tests/integer-warn.errwarn
EXTRA_DIST += modules/objfmts/bin/tests/integer.asm
EXTRA_DIST += modules/objfmts/bin/tests/integer.hex
EXTRA_DIST += modules/objfmts/bin/tests/levelop.asm
EXTRA_DIST += modules/objfmts/bin/tests/levelop.hex
EXTRA_DIST += modules/objfmts/bin/tests/reserve.asm
EXTRA_DIST += modules/objfmts/bin/tests/reserve.hex
EXTRA_DIST += modules/objfmts/bin/tests/reserve.errwarn
EXTRA_DIST += modules/objfmts/bin/tests/shr.asm
EXTRA_DIST += modules/objfmts/bin/tests/shr.hex

EXTRA_DIST += modules/objfmts/bin/tests/multisect/Makefile.inc

include modules/objfmts/bin/tests/multisect/Makefile.inc
