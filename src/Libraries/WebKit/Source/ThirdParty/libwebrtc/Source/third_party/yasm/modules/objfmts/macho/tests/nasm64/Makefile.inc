/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

TESTS += modules/objfmts/macho/tests/nasm64/macho64_test.sh

EXTRA_DIST += modules/objfmts/macho/tests/nasm64/nasm-macho64-pic.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/nasm-macho64-pic.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/macho64_test.sh
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/machotest64.c
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/machotest64.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/machotest64.hex
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/macho-reloc64-err.asm
EXTRA_DIST += modules/objfmts/macho/tests/nasm64/macho-reloc64-err.errwarn
