/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#include <xlocale.h>
#include <sys/cdefs.h>

#if !defined(__BIONIC_STDLIB_INLINE)
#define __BIONIC_STDLIB_INLINE static __inline
#endif

__BEGIN_DECLS

__BIONIC_STDLIB_INLINE double strtod_l(const char* _Nonnull __s, char* _Nullable * _Nullable __end_ptr, locale_t _Nonnull __l) {
  return strtod(__s, __end_ptr);
}

__BIONIC_STDLIB_INLINE float strtof_l(const char* _Nonnull __s, char* _Nullable * _Nullable __end_ptr, locale_t _Nonnull __l) {
  return strtof(__s, __end_ptr);
}

__END_DECLS
