/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
 * NASM version strings, defined in ver.c
 */

#ifndef NASM_VER_H
#define NASM_VER_H

#include "compiler.h"

extern const char nasm_version[];
extern const char nasm_date[];
extern const char nasm_compile_options[];

extern bool reproducible;

extern const char * pure_func nasm_comment(void);
extern size_t pure_func nasm_comment_len(void);

extern const char * pure_func nasm_signature(void);
extern size_t pure_func nasm_signature_len(void);

#endif /* NASM_VER_H */
