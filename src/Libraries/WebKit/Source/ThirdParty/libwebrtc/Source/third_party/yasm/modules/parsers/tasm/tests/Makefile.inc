/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

TESTS += modules/parsers/tasm/tests/tasm_test.sh

EXTRA_DIST += modules/parsers/tasm/tests/tasm_test.sh
EXTRA_DIST += modules/parsers/tasm/tests/array.asm
EXTRA_DIST += modules/parsers/tasm/tests/array.hex
EXTRA_DIST += modules/parsers/tasm/tests/case.asm
EXTRA_DIST += modules/parsers/tasm/tests/case.hex
EXTRA_DIST += modules/parsers/tasm/tests/charstr.asm
EXTRA_DIST += modules/parsers/tasm/tests/charstr.hex
EXTRA_DIST += modules/parsers/tasm/tests/dup.asm
EXTRA_DIST += modules/parsers/tasm/tests/dup.hex
EXTRA_DIST += modules/parsers/tasm/tests/equal.asm
EXTRA_DIST += modules/parsers/tasm/tests/equal.hex
EXTRA_DIST += modules/parsers/tasm/tests/expr.asm
EXTRA_DIST += modules/parsers/tasm/tests/expr.hex
EXTRA_DIST += modules/parsers/tasm/tests/irp.asm
EXTRA_DIST += modules/parsers/tasm/tests/irp.hex
EXTRA_DIST += modules/parsers/tasm/tests/label.asm
EXTRA_DIST += modules/parsers/tasm/tests/label.hex
EXTRA_DIST += modules/parsers/tasm/tests/les.asm
EXTRA_DIST += modules/parsers/tasm/tests/les.hex
EXTRA_DIST += modules/parsers/tasm/tests/lidt.asm
EXTRA_DIST += modules/parsers/tasm/tests/lidt.hex
EXTRA_DIST += modules/parsers/tasm/tests/macro.asm
EXTRA_DIST += modules/parsers/tasm/tests/macro.hex
EXTRA_DIST += modules/parsers/tasm/tests/offset.asm
EXTRA_DIST += modules/parsers/tasm/tests/offset.hex
EXTRA_DIST += modules/parsers/tasm/tests/quote.asm
EXTRA_DIST += modules/parsers/tasm/tests/quote.hex
EXTRA_DIST += modules/parsers/tasm/tests/res.asm
EXTRA_DIST += modules/parsers/tasm/tests/res.errwarn
EXTRA_DIST += modules/parsers/tasm/tests/res.hex
EXTRA_DIST += modules/parsers/tasm/tests/segment.asm
EXTRA_DIST += modules/parsers/tasm/tests/segment.hex
EXTRA_DIST += modules/parsers/tasm/tests/size.asm
EXTRA_DIST += modules/parsers/tasm/tests/size.hex
EXTRA_DIST += modules/parsers/tasm/tests/struc.asm
EXTRA_DIST += modules/parsers/tasm/tests/struc.errwarn
EXTRA_DIST += modules/parsers/tasm/tests/struc.hex

EXTRA_DIST += modules/parsers/tasm/tests/exe/Makefile.inc

include modules/parsers/tasm/tests/exe/Makefile.inc
