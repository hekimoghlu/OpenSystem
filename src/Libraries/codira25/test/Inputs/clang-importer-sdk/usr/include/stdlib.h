/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#if defined(_WIN32) || defined(WIN32)
#define SDK_STDLIB_H
#endif

#ifndef SDK_STDLIB_H
#define SDK_STDLIB_H

#include <stdint.h>

typedef long ldiv_t;
typedef long long lldiv_t;

int posix_memalign(void **, size_t, size_t);
void free(void *);

ldiv_t ldiv(long int, long int);
lldiv_t lldiv(long long int, long long int);

_Noreturn void abort(void);

#endif // SDK_STDLIB_H
