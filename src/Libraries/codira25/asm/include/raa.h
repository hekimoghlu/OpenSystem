/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#ifndef NASM_RAA_H
#define NASM_RAA_H 1

#include "compiler.h"

struct RAA;
typedef uint64_t raaindex;

#define raa_init() NULL
void raa_free(struct RAA *);
int64_t pure_func raa_read(struct RAA *, raaindex);
void * pure_func raa_read_ptr(struct RAA *, raaindex);
struct RAA * never_null raa_write(struct RAA *r, raaindex posn, int64_t value);
struct RAA * never_null raa_write_ptr(struct RAA *r, raaindex posn, void *value);

#endif                          /* NASM_RAA_H */
