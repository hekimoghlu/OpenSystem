/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

libyasm_a_SOURCES += modules/objfmts/elf/elf.c
libyasm_a_SOURCES += modules/objfmts/elf/elf.h
libyasm_a_SOURCES += modules/objfmts/elf/elf-objfmt.c
libyasm_a_SOURCES += modules/objfmts/elf/elf-machine.h
libyasm_a_SOURCES += modules/objfmts/elf/elf-x86-x86.c
libyasm_a_SOURCES += modules/objfmts/elf/elf-x86-amd64.c
libyasm_a_SOURCES += modules/objfmts/elf/elf-x86-x32.c

YASM_MODULES += objfmt_elf objfmt_elf32 objfmt_elf64 objfmt_elfx32

EXTRA_DIST += modules/objfmts/elf/tests/Makefile.inc

include modules/objfmts/elf/tests/Makefile.inc
