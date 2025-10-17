/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
#ifndef _PLATFORM_COMPAT_H_
#define _PLATFORM_COMPAT_H_

#include <platform/string.h>

/* Compat macros for primitives */
#define bzero            _platform_bzero
#define memchr           _platform_memchr
#define memcmp           _platform_memcmp
#define memmove          _platform_memmove
#define memccpy          _platform_memccpy
#define memset           _platform_memset
#define memset_pattern4  _platform_memset_pattern4
#define memset_pattern8  _platform_memset_pattern8
#define memset_pattern16 _platform_memset_pattern16
#define strchr           _platform_strchr
#define strcmp           _platform_strcmp
#define strcpy           _platform_strcpy
#define strlcat          _platform_strlcat
#define strlcpy          _platform_strlcpy
#define strlen           _platform_strlen
#define strncmp          _platform_strncmp
#define strncpy          _platform_strncpy
#define strnlen          _platform_strnlen
#define strstr           _platform_strstr

#endif /* _PLATFORM_COMPAT_H_ */
