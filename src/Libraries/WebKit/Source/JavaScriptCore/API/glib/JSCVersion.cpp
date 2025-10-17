/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#include "JSCVersion.h"

/**
 * JSCVersion:
 * @Short_description: Provides the JavaScriptCore version
 * @Title: JSCVersion
 *
 * Provides convenience functions returning JavaScriptCore's major, minor and
 * micro versions of the JavaScriptCore library your code is running
 * against. This is not necessarily the same as the
 * #JSC_MAJOR_VERSION, #JSC_MINOR_VERSION or
 * #JSC_MICRO_VERSION, which represent the version of the JavaScriptCore
 * headers included when compiling the code.
 *
 */

/**
 * jsc_get_major_version:
 *
 * Returns the major version number of the JavaScriptCore library.
 * (e.g. in JavaScriptCore version 1.8.3 this is 1.)
 *
 * This function is in the library, so it represents the JavaScriptCore library
 * your code is running against. Contrast with the #JSC_MAJOR_VERSION
 * macro, which represents the major version of the JavaScriptCore headers you
 * have included when compiling your code.
 *
 * Returns: the major version number of the JavaScriptCore library
 */
guint jsc_get_major_version(void)
{
    return JSC_MAJOR_VERSION;
}

/**
 * jsc_get_minor_version:
 *
 * Returns the minor version number of the JavaScriptCore library.
 * (e.g. in JavaScriptCore version 1.8.3 this is 8.)
 *
 * This function is in the library, so it represents the JavaScriptCore library
 * your code is running against. Contrast with the #JSC_MINOR_VERSION
 * macro, which represents the minor version of the JavaScriptCore headers you
 * have included when compiling your code.
 *
 * Returns: the minor version number of the JavaScriptCore library
 */
guint jsc_get_minor_version(void)
{
    return JSC_MINOR_VERSION;
}

/**
 * jsc_get_micro_version:
 *
 * Returns the micro version number of the JavaScriptCore library.
 * (e.g. in JavaScriptCore version 1.8.3 this is 3.)
 *
 * This function is in the library, so it represents the JavaScriptCore library
 * your code is running against. Contrast with the #JSC_MICRO_VERSION
 * macro, which represents the micro version of the JavaScriptCore headers you
 * have included when compiling your code.
 *
 * Returns: the micro version number of the JavaScriptCore library
 */
guint jsc_get_micro_version(void)
{
    return JSC_MICRO_VERSION;
}
