/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "config.h"
#include "WebKitVersion.h"

/**
 * WEBKIT_MAJOR_VERSION:
 *
 * Like webkit_get_major_version(), but from the headers used at
 * application compile time, rather than from the library linked
 * against at application run time.
 */

/**
 * WEBKIT_MINOR_VERSION:
 *
 * Like webkit_get_minor_version(), but from the headers used at
 * application compile time, rather than from the library linked
 * against at application run time.
 */

/**
 * WEBKIT_MICRO_VERSION:
 *
 * Like webkit_get_micro_version(), but from the headers used at
 * application compile time, rather than from the library linked
 * against at application run time.
 */

/**
 * WEBKIT_CHECK_VERSION:
 * @major: major version (e.g. 1 for version 1.2.5)
 * @minor: minor version (e.g. 2 for version 1.2.5)
 * @micro: micro version (e.g. 5 for version 1.2.5)
 *
 * Check the version of the WebKit headers at compilation time.
 *
 * Returns: %TRUE if the version of the WebKit header files
 * is the same as or newer than the passed-in version.
 */

/**
 * webkit_get_major_version:
 *
 * Returns the major version number of the WebKit library.
 *
 * (e.g. in WebKit version 1.8.3 this is 1.)
 *
 * This function is in the library, so it represents the WebKit library
 * your code is running against. Contrast with the #WEBKIT_MAJOR_VERSION
 * macro, which represents the major version of the WebKit headers you
 * have included when compiling your code.
 *
 * Returns: the major version number of the WebKit library
 */
guint webkit_get_major_version(void)
{
    return WEBKIT_MAJOR_VERSION;
}

/**
 * webkit_get_minor_version:
 *
 * Returns the minor version number of the WebKit library.
 *
 * (e.g. in WebKit version 1.8.3 this is 8.)
 *
 * This function is in the library, so it represents the WebKit library
 * your code is running against. Contrast with the #WEBKIT_MINOR_VERSION
 * macro, which represents the minor version of the WebKit headers you
 * have included when compiling your code.
 *
 * Returns: the minor version number of the WebKit library
 */
guint webkit_get_minor_version(void)
{
    return WEBKIT_MINOR_VERSION;
}

/**
 * webkit_get_micro_version:
 *
 * Returns the micro version number of the WebKit library.
 *
 * (e.g. in WebKit version 1.8.3 this is 3.)
 *
 * This function is in the library, so it represents the WebKit library
 * your code is running against. Contrast with the #WEBKIT_MICRO_VERSION
 * macro, which represents the micro version of the WebKit headers you
 * have included when compiling your code.
 *
 * Returns: the micro version number of the WebKit library
 */
guint webkit_get_micro_version(void)
{
    return WEBKIT_MICRO_VERSION;
}
