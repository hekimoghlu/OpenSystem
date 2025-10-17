/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "WPEVersion.h"

/**
 * WPEVersion:
 * @Short_description: Provides the WPE platform version
 * @Title: WPEVersion
 *
 * Provides convenience functions returning WPE's major, minor and
 * micro versions of the WPE platform library your code is running
 * against. This is not necessarily the same as the
 * #WPE_PLATFORM_MAJOR_VERSION, #WPE_PLATFORM_MINOR_VERSION or
 * #WPE_PLATFORM_MICRO_VERSION, which represent the version of the WPE platform
 * headers included when compiling the code.
 */

/**
 * wpe_platform_get_major_version:
 *
 * Returns the major version number of the WPE platform library.
 * (e.g. in WPEPlatform version 1.8.3 this is 1.)
 *
 * This function is in the library, so it represents the WPE platform library
 * your code is running against. Contrast with the #WPE_PLATFORM_MAJOR_VERSION
 * macro, which represents the major version of the WPE platform headers you
 * have included when compiling your code.
 *
 * Returns: the major version number of the WPE platform library
 */
guint wpe_platform_get_major_version(void)
{
    return WPE_PLATFORM_MAJOR_VERSION;
}

/**
 * wpe_platform_get_minor_version:
 *
 * Returns the minor version number of the WPE platform library.
 * (e.g. in WPEPlatform version 1.8.3 this is 8.)
 *
 * This function is in the library, so it represents the WPE platform library
 * your code is running against. Contrast with the #WPE_PLATFORM_MINOR_VERSION
 * macro, which represents the minor version of the WPE platform headers you
 * have included when compiling your code.
 *
 * Returns: the minor version number of the WPE platform library
 */
guint wpe_platform_get_minor_version(void)
{
    return WPE_PLATFORM_MINOR_VERSION;
}

/**
 * wpe_platform_get_micro_version:
 *
 * Returns the micro version number of the WPE platform library.
 * (e.g. in WPEPlatform version 1.8.3 this is 3.)
 *
 * This function is in the library, so it represents the WPE platform library
 * your code is running against. Contrast with the #WPE_PLATFORM_MICRO_VERSION
 * macro, which represents the micro version of the WPE platform headers you
 * have included when compiling your code.
 *
 * Returns: the micro version number of the WPE platform library
 */
guint wpe_platform_get_micro_version(void)
{
    return WPE_PLATFORM_MICRO_VERSION;
}
