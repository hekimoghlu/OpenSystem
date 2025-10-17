/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#pragma once

#include <stddef.h>

#define __attribute_pure__ __attribute__((__pure__))

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
    #define TEST_CONST_RETURN  const
#else
    #define TEST_CONST_RETURN
#endif


void* _Nonnull memcpy(void* _Nonnull, const void* _Nonnull, size_t);

void* _Nonnull memcpy42(void* _Nonnull, const void* _Nonnull, size_t);

void TEST_CONST_RETURN* _Nullable memchr(const void* _Nonnull __s, int __ch, size_t __n) __attribute_pure__;

void* _Nonnull memmove(void* _Nonnull __dst, const void* _Nonnull __src, size_t __n);

void* _Nonnull memset(void* _Nonnull __dst, int __ch, size_t __n);

char TEST_CONST_RETURN* strrchr(const char* __s, int __ch) __attribute_pure__;

char* _Nonnull strcpy(char* _Nonnull __dst, const char* _Nonnull __src);
char* _Nonnull strcat(char* _Nonnull __dst, const char* _Nonnull __src);

#ifdef __cplusplus
}
#endif
