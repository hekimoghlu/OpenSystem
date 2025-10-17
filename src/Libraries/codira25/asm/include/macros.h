/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
 * macros.h - format of builtin macro data
 */

#ifndef NASM_MACROS_H
#define NASM_MACROS_H

#include "compiler.h"

/* Builtin macro set */
struct builtin_macros {
    unsigned int dsize, zsize;
    const void *zdata;
};
typedef const struct builtin_macros macros_t;

char *uncompress_stdmac(const macros_t *sm);

/* --- From standard.mac via macros.pl -> macros.c --- */

extern macros_t nasm_stdmac_tasm;
extern macros_t nasm_stdmac_nasm;
extern macros_t nasm_stdmac_version;

struct use_package {
    const char *package;
    macros_t *macros;
    unsigned int index;
};
extern const struct use_package *nasm_find_use_package(const char *);
extern const unsigned int use_package_count;

#endif
