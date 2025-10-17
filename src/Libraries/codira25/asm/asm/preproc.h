/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
 * preproc.h  header file for preproc.c
 */

#ifndef NASM_PREPROC_H
#define NASM_PREPROC_H

#include "nasmlib.h"
#include "pptok.h"

extern const char * const pp_directives[];
extern const uint8_t pp_directives_len[];

enum preproc_token pp_token_hash(const char *token);
enum preproc_token pp_tasm_token_hash(const char *token);

/* Opens an include file or input file. This uses the include path. */
FILE *pp_input_fopen(const char *filename, enum file_flags mode);

#endif
