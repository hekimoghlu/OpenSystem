/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
/*
 * nasmlib.c	library routines for the Netwide Assembler
 */

#include "compiler.h"
#include "nasmlib.h"
#include "error.h"

/*
 * Add/modify a filename extension, assumed to be a period-delimited
 * field at the very end of the filename.  Returns a newly allocated
 * string buffer.
 */
const char *filename_set_extension(const char *inname, const char *extension)
{
    const char *q = inname;
    char *p;
    size_t elen = strlen(extension);
    size_t baselen;

    q = strrchrnul(inname, '.');   /* find extension or end of string */
    baselen = q - inname;

    p = nasm_malloc(baselen + elen + 1);

    memcpy(p, inname, baselen);
    memcpy(p+baselen, extension, elen+1);

    return p;
}
