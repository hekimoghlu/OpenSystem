/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#ifndef __MALLOC_COMMON_H
#define __MALLOC_COMMON_H

MALLOC_NOEXPORT
const char *
malloc_common_strstr(const char *src, const char *target, size_t target_len);

MALLOC_NOEXPORT
long
malloc_common_convert_to_long(const char *ptr, const char **end_ptr);

MALLOC_NOEXPORT
const char *
malloc_common_value_for_key(const char *src, const char *key);

MALLOC_NOEXPORT
const char *
malloc_common_value_for_key_copy(const char *src, const char *key,
		 char *bufp, size_t maxlen);

#endif // __MALLOC_COMMON_H
