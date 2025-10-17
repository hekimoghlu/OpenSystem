/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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

TESTS += modules/parsers/gas/tests/bin/gas_bin_test.sh

EXTRA_DIST += modules/parsers/gas/tests/bin/gas_bin_test.sh
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-comment.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-comment.errwarn
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-comment.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-intel_syntax-noprefix.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-intel_syntax-noprefix.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-llabel.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-llabel.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-macro.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-macro.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-set.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-set.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-str.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/gas-str.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/rept-err.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/rept-err.errwarn
EXTRA_DIST += modules/parsers/gas/tests/bin/reptempty.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/reptempty.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/reptlong.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/reptlong.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/reptnested.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/reptnested.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/reptsimple.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/reptsimple.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/reptwarn.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/reptwarn.errwarn
EXTRA_DIST += modules/parsers/gas/tests/bin/reptwarn.hex
EXTRA_DIST += modules/parsers/gas/tests/bin/reptzero.asm
EXTRA_DIST += modules/parsers/gas/tests/bin/reptzero.hex

