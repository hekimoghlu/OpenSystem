/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

TESTS += modules/objfmts/macho/tests/nasm32/macho32_test.sh

EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32_test.sh
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/machotest.c
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/machotest.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/machotest.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho-reloc.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho-reloc.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-pext.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-pext.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-pic.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-pic.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-sect.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-sect.errwarn
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-sect.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-size.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm32/macho32-size.hex
