/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#include "WebKitApplicationInfo.h"

#include <wtf/text/CString.h>

/**
 * WebKitApplicationInfo: (ref-func webkit_application_info_ref) (unref-func webkit_application_info_unref)
 *
 * Information about an application running in automation mode.
 */

struct _WebKitApplicationInfo {
    CString name;
    uint64_t majorVersion;
    uint64_t minorVersion;
    uint64_t microVersion;

    int referenceCount { 1 };
};

G_DEFINE_BOXED_TYPE(WebKitApplicationInfo, webkit_application_info, webkit_application_info_ref, webkit_application_info_unref)

/**
 * webkit_application_info_new: (constructor)
 *
 * Creates a new #WebKitApplicationInfo
 *
 * Returns: (transfer full): the newly created #WebKitApplicationInfo.
 *
 * since: 2.18
 */
WebKitApplicationInfo* webkit_application_info_new()
{
    WebKitApplicationInfo* info = static_cast<WebKitApplicationInfo*>(fastMalloc(sizeof(WebKitApplicationInfo)));
    new (info) WebKitApplicationInfo();
    return info;
}

/**
 * webkit_application_info_ref:
 * @info: a #WebKitApplicationInfo
 *
 * Atomically increments the reference count of @info by one.
 *
 * This
 * function is MT-safe and may be called from any thread.
 *
 * Returns: The passed in #WebKitApplicationInfo
 *
 * Since: 2.18
 */
WebKitApplicationInfo* webkit_application_info_ref(WebKitApplicationInfo* info)
{
    g_atomic_int_inc(&info->referenceCount);
    return info;
}

/**
 * webkit_application_info_unref:
 * @info: a #WebKitApplicationInfo
 *
 * Atomically decrements the reference count of @info by one.
 *
 * If the
 * reference count drops to 0, all memory allocated by the #WebKitApplicationInfo is
 * released. This function is MT-safe and may be called from any
 * thread.
 *
 * Since: 2.18
 */
void webkit_application_info_unref(WebKitApplicationInfo* info)
{
    if (g_atomic_int_dec_and_test(&info->referenceCount)) {
        info->~WebKitApplicationInfo();
        fastFree(info);
    }
}

/**
 * webkit_application_info_set_name:
 * @info: a #WebKitApplicationInfo
 * @name: the application name
 *
 * Set the name of the application.
 *
 * If not provided, or %NULL is passed,
 * g_get_prgname() will be used.
 *
 * Since: 2.18
 */
void webkit_application_info_set_name(WebKitApplicationInfo* info, const char* name)
{
    g_return_if_fail(info);

    info->name = name;
}

/**
 * webkit_application_info_get_name:
 * @info: a #WebKitApplicationInfo
 *
 * Get the name of the application.
 *
 * If webkit_application_info_set_name() hasn't been
 * called with a valid name, this returns g_get_prgname().
 *
 * Returns: the application name
 *
 * Since: 2.18
 */
const char* webkit_application_info_get_name(WebKitApplicationInfo* info)
{
    g_return_val_if_fail(info, nullptr);

    if (!info->name.isNull())
        return info->name.data();

    return g_get_prgname();
}

/**
 * webkit_application_info_set_version:
 * @info: a #WebKitApplicationInfo
 * @major: the major version number
 * @minor: the minor version number
 * @micro: the micro version number
 *
 * Set the application version.
 *
 * If the application doesn't use the format
 * major.minor.micro you can pass 0 as the micro to use major.minor, or pass
 * 0 as both micro and minor to use only major number. Any other format must
 * be converted to major.minor.micro so that it can be used in version comparisons.
 *
 * Since: 2.18
 */
void webkit_application_info_set_version(WebKitApplicationInfo* info, guint64 major, guint64 minor, guint64 micro)
{
    g_return_if_fail(info);

    info->majorVersion = major;
    info->minorVersion = minor;
    info->microVersion = micro;
}

/**
 * webkit_application_info_get_version:
 * @info: a #WebKitApplicationInfo
 * @major: (out): return location for the major version number
 * @minor: (out) (allow-none): return location for the minor version number
 * @micro: (out) (allow-none): return location for the micro version number
 *
 * Get the application version previously set with webkit_application_info_set_version().
 *
 * Since: 2.18
 */
void webkit_application_info_get_version(WebKitApplicationInfo* info, guint64* major, guint64* minor, guint64* micro)
{
    g_return_if_fail(info && major);

    *major = info->majorVersion;
    if (minor)
        *minor = info->minorVersion;
    if (micro)
        *micro = info->microVersion;
}
