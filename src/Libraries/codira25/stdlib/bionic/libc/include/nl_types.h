/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

/**
 * @file nl_types.h
 * @brief Message catalogs.
 *
 * Android offers a no-op implementation of these functions to ease porting of historical software.
 */

#include <sys/cdefs.h>

__BEGIN_DECLS

/**
 * catopen() flag to use the current locale.
 */
#define NL_CAT_LOCALE 1

/**
 * catgets() default set number.
 */
#define NL_SETD 1

/** Message catalog type. */
typedef void* nl_catd;

/** The type of the constants in `<langinfo.h>`, used by nl_langinfo(). */
typedef int nl_item;

/**
 * [catopen(3)](https://man7.org/linux/man-pages/man3/catopen.3.html) opens a message catalog.
 *
 * On Android, this always returns failure: `((nl_catd) -1)`.
 *
 * Available since API level 28.
 */

#if __BIONIC_AVAILABILITY_GUARD(26)
nl_catd _Nonnull catopen(const char* _Nonnull __name, int __flag) __INTRODUCED_IN(26);

/**
 * [catgets(3)](https://man7.org/linux/man-pages/man3/catgets.3.html) translates the given message
 * using the given message catalog.
 *
 * On Android, this always returns `__msg`.
 *
 * Available since API level 28.
 */
char* _Nonnull catgets(nl_catd _Nonnull __catalog, int __set_number, int __msg_number, const char* _Nonnull __msg) __INTRODUCED_IN(26);

/**
 * [catclose(3)](https://man7.org/linux/man-pages/man3/catclose.3.html) closes a message catalog.
 *
 * On Android, this always returns -1 with `errno` set to `EBADF`.
 */
int catclose(nl_catd _Nonnull __catalog) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


__END_DECLS
