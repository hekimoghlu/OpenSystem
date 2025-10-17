/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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

mode_t __umask_chk(mode_t);
mode_t __umask_real(mode_t mode) __RENAME(umask);

#if defined(__BIONIC_FORTIFY)

/* Abuse enable_if to make this an overload of umask. */
__BIONIC_FORTIFY_INLINE
mode_t umask(mode_t mode)
    __overloadable
    __enable_if(1, "")
    __clang_error_if(mode & ~0777, "'umask' called with invalid mode") {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
  return __umask_chk(mode);
#else
  return __umask_real(mode);
#endif
}

#endif /* defined(__BIONIC_FORTIFY) */
