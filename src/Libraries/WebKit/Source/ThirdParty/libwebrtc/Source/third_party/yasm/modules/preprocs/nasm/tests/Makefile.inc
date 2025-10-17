/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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

TESTS += modules/preprocs/nasm/tests/nasmpp_test.sh

EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp_test.sh
EXTRA_DIST += modules/preprocs/nasm/tests/16args.asm
EXTRA_DIST += modules/preprocs/nasm/tests/16args.hex
EXTRA_DIST += modules/preprocs/nasm/tests/ifcritical-err.asm
EXTRA_DIST += modules/preprocs/nasm/tests/ifcritical-err.errwarn
EXTRA_DIST += modules/preprocs/nasm/tests/longline.asm
EXTRA_DIST += modules/preprocs/nasm/tests/longline.hex
EXTRA_DIST += modules/preprocs/nasm/tests/macroeof-err.asm
EXTRA_DIST += modules/preprocs/nasm/tests/macroeof-err.errwarn
EXTRA_DIST += modules/preprocs/nasm/tests/noinclude-err.asm
EXTRA_DIST += modules/preprocs/nasm/tests/noinclude-err.errwarn
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-bigint.asm
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-bigint.hex
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-decimal.asm
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-decimal.hex
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-nested.asm
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-nested.errwarn
EXTRA_DIST += modules/preprocs/nasm/tests/nasmpp-nested.hex
EXTRA_DIST += modules/preprocs/nasm/tests/orgsect.asm
EXTRA_DIST += modules/preprocs/nasm/tests/orgsect.hex
EXTRA_DIST += modules/preprocs/nasm/tests/scope-err.asm
EXTRA_DIST += modules/preprocs/nasm/tests/scope-err.errwarn
