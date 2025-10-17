/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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

#include <sys/cdefs.h>

__BEGIN_DECLS

// PAGE_SIZE is going away in Android. Prefer getpagesize() instead.
//
// For more info, see https://developer.android.com/16kb-page-size.
//
// To restore the original behavior, use __BIONIC_DEPRECATED_PAGE_SIZE_MACRO.

#if (defined(__NDK_MAJOR__) && __NDK_MAJOR__ <= 27 && !defined(__BIONIC_NO_PAGE_SIZE_MACRO)) \
    || defined(__BIONIC_DEPRECATED_PAGE_SIZE_MACRO) \
    || defined(__arm__) \
    || defined(__i386__)
#define PAGE_SIZE 4096
#define PAGE_MASK (~(PAGE_SIZE - 1))
#endif

__END_DECLS
