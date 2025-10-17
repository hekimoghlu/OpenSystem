/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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

TESTS += modules/parsers/gas/tests/gas_test.sh

EXTRA_DIST += modules/parsers/gas/tests/gas_test.sh
EXTRA_DIST += modules/parsers/gas/tests/dataref-imm.asm
EXTRA_DIST += modules/parsers/gas/tests/dataref-imm.hex
EXTRA_DIST += modules/parsers/gas/tests/datavis.asm
EXTRA_DIST += modules/parsers/gas/tests/datavis.errwarn
EXTRA_DIST += modules/parsers/gas/tests/datavis.hex
EXTRA_DIST += modules/parsers/gas/tests/datavis2.asm
EXTRA_DIST += modules/parsers/gas/tests/datavis2.hex
EXTRA_DIST += modules/parsers/gas/tests/execsect.asm
EXTRA_DIST += modules/parsers/gas/tests/execsect.hex
EXTRA_DIST += modules/parsers/gas/tests/gas-fill.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-fill.hex
EXTRA_DIST += modules/parsers/gas/tests/gas-float.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-float.hex
EXTRA_DIST += modules/parsers/gas/tests/gas-instlabel.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-instlabel.hex
EXTRA_DIST += modules/parsers/gas/tests/gas-line-err.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-line-err.errwarn
EXTRA_DIST += modules/parsers/gas/tests/gas-line2-err.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-line2-err.errwarn
EXTRA_DIST += modules/parsers/gas/tests/gas-push.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-push.hex
EXTRA_DIST += modules/parsers/gas/tests/gas-segprefix.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-segprefix.hex
EXTRA_DIST += modules/parsers/gas/tests/gas-semi.asm
EXTRA_DIST += modules/parsers/gas/tests/gas-semi.hex
EXTRA_DIST += modules/parsers/gas/tests/gassectalign.asm
EXTRA_DIST += modules/parsers/gas/tests/gassectalign.hex
EXTRA_DIST += modules/parsers/gas/tests/jmpcall.asm
EXTRA_DIST += modules/parsers/gas/tests/jmpcall.errwarn
EXTRA_DIST += modules/parsers/gas/tests/jmpcall.hex
EXTRA_DIST += modules/parsers/gas/tests/leb128.asm
EXTRA_DIST += modules/parsers/gas/tests/leb128.hex
EXTRA_DIST += modules/parsers/gas/tests/localcomm.asm
EXTRA_DIST += modules/parsers/gas/tests/localcomm.hex
EXTRA_DIST += modules/parsers/gas/tests/reggroup-err.asm
EXTRA_DIST += modules/parsers/gas/tests/reggroup-err.errwarn
EXTRA_DIST += modules/parsers/gas/tests/reggroup.asm
EXTRA_DIST += modules/parsers/gas/tests/reggroup.hex
EXTRA_DIST += modules/parsers/gas/tests/strzero.asm
EXTRA_DIST += modules/parsers/gas/tests/strzero.hex
EXTRA_DIST += modules/parsers/gas/tests/varinsn.asm
EXTRA_DIST += modules/parsers/gas/tests/varinsn.hex

EXTRA_DIST += modules/parsers/gas/tests/bin/Makefile.inc
include modules/parsers/gas/tests/bin/Makefile.inc
