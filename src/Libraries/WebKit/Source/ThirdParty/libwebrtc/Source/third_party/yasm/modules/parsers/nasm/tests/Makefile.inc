/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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

TESTS += modules/parsers/nasm/tests/nasm_test.sh

EXTRA_DIST += modules/parsers/nasm/tests/nasm_test.sh
EXTRA_DIST += modules/parsers/nasm/tests/alignnop16.asm
EXTRA_DIST += modules/parsers/nasm/tests/alignnop16.hex
EXTRA_DIST += modules/parsers/nasm/tests/alignnop32.asm
EXTRA_DIST += modules/parsers/nasm/tests/alignnop32.hex
EXTRA_DIST += modules/parsers/nasm/tests/charconstmath.asm
EXTRA_DIST += modules/parsers/nasm/tests/charconstmath.hex
EXTRA_DIST += modules/parsers/nasm/tests/dirwarning.asm
EXTRA_DIST += modules/parsers/nasm/tests/dirwarning.errwarn
EXTRA_DIST += modules/parsers/nasm/tests/dirwarning.hex
EXTRA_DIST += modules/parsers/nasm/tests/dy.asm
EXTRA_DIST += modules/parsers/nasm/tests/dy.hex
EXTRA_DIST += modules/parsers/nasm/tests/endcomma.asm
EXTRA_DIST += modules/parsers/nasm/tests/endcomma.hex
EXTRA_DIST += modules/parsers/nasm/tests/equcolon.asm
EXTRA_DIST += modules/parsers/nasm/tests/equcolon.hex
EXTRA_DIST += modules/parsers/nasm/tests/equlocal.asm
EXTRA_DIST += modules/parsers/nasm/tests/equlocal.hex
EXTRA_DIST += modules/parsers/nasm/tests/hexconst.asm
EXTRA_DIST += modules/parsers/nasm/tests/hexconst.hex
EXTRA_DIST += modules/parsers/nasm/tests/long.asm
EXTRA_DIST += modules/parsers/nasm/tests/long.hex
EXTRA_DIST += modules/parsers/nasm/tests/locallabel.asm
EXTRA_DIST += modules/parsers/nasm/tests/locallabel.hex
EXTRA_DIST += modules/parsers/nasm/tests/locallabel2.asm
EXTRA_DIST += modules/parsers/nasm/tests/locallabel2.hex
EXTRA_DIST += modules/parsers/nasm/tests/nasm-prefix.asm
EXTRA_DIST += modules/parsers/nasm/tests/nasm-prefix.hex
EXTRA_DIST += modules/parsers/nasm/tests/newsect.asm
EXTRA_DIST += modules/parsers/nasm/tests/newsect.hex
EXTRA_DIST += modules/parsers/nasm/tests/orphannowarn.asm
EXTRA_DIST += modules/parsers/nasm/tests/orphannowarn.hex
EXTRA_DIST += modules/parsers/nasm/tests/prevlocalwarn.asm
EXTRA_DIST += modules/parsers/nasm/tests/prevlocalwarn.errwarn
EXTRA_DIST += modules/parsers/nasm/tests/prevlocalwarn.hex
EXTRA_DIST += modules/parsers/nasm/tests/strucalign.asm
EXTRA_DIST += modules/parsers/nasm/tests/strucalign.hex
EXTRA_DIST += modules/parsers/nasm/tests/struczero.asm
EXTRA_DIST += modules/parsers/nasm/tests/struczero.hex
EXTRA_DIST += modules/parsers/nasm/tests/strucbase.asm
EXTRA_DIST += modules/parsers/nasm/tests/strucbase.hex
EXTRA_DIST += modules/parsers/nasm/tests/syntax-err.asm
EXTRA_DIST += modules/parsers/nasm/tests/syntax-err.errwarn
EXTRA_DIST += modules/parsers/nasm/tests/uscore.asm
EXTRA_DIST += modules/parsers/nasm/tests/uscore.hex

EXTRA_DIST += modules/parsers/nasm/tests/worphan/Makefile.inc

include modules/parsers/nasm/tests/worphan/Makefile.inc
