/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
#include "ver.h"
#include "version.h"

/* This is printed when entering nasm -v */
const char nasm_version[] = NASM_VER;
const char nasm_date[] = __DATE__;
const char nasm_compile_options[] = ""
#ifdef DEBUG
    " with -DDEBUG"
#endif
    ;

bool reproducible;              /* Reproducible output */

/* These are used by some backends. For a reproducible build,
 * these cannot contain version numbers.
 */
static const char * const _nasm_comment[2] =
{
    "The Netwide Assembler " NASM_VER,
    "The Netwide Assembler"
};

static const char * const _nasm_signature[2] = {
    "NASM " NASM_VER,
    "NASM"
};

const char * pure_func nasm_comment(void)
{
    return _nasm_comment[reproducible];
}

size_t pure_func nasm_comment_len(void)
{
    return strlen(nasm_comment());
}

const char * pure_func nasm_signature(void)
{
    return _nasm_signature[reproducible];
}

size_t pure_func nasm_signature_len(void)
{
    return strlen(nasm_signature());
}
