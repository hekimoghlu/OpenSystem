/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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

TESTS += modules/objfmts/elf/tests/amd64/elf_amd64_test.sh

EXTRA_DIST += modules/objfmts/elf/tests/amd64/elf_amd64_test.sh
EXTRA_DIST += modules/objfmts/elf/tests/amd64/elf-rip.asm
EXTRA_DIST += modules/objfmts/elf/tests/amd64/elf-rip.hex
EXTRA_DIST += modules/objfmts/elf/tests/amd64/elfso64.asm
EXTRA_DIST += modules/objfmts/elf/tests/amd64/elfso64.hex
EXTRA_DIST += modules/objfmts/elf/tests/amd64/gotpcrel.asm
EXTRA_DIST += modules/objfmts/elf/tests/amd64/gotpcrel.hex
EXTRA_DIST += modules/objfmts/elf/tests/amd64/multiplefixup.asm
EXTRA_DIST += modules/objfmts/elf/tests/amd64/multiplefixup.hex
