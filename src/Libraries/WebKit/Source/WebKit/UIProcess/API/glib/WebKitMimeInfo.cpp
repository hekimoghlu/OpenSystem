/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "WebKitMimeInfo.h"

/**
 * WebKitMimeInfo: (ref-func webkit_mime_info_ref) (unref-func webkit_mime_info_unref)
 *
 * Information about a MIME type.
 */

struct _WebKitMimeInfo {
};

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
G_DEFINE_BOXED_TYPE(WebKitMimeInfo, webkit_mime_info, webkit_mime_info_ref, webkit_mime_info_unref)
ALLOW_DEPRECATED_DECLARATIONS_END

/**
 * webkit_mime_info_ref:
 * @info: a #WebKitMimeInfo
 *
 * Atomically increments the reference count of @info by one.
 *
 * This function is MT-safe and may be called from any thread.
 *
 * Returns: The passed in #WebKitMimeInfo
 *
 * Deprecated: 2.32
 */
WebKitMimeInfo* webkit_mime_info_ref(WebKitMimeInfo*)
{
    return nullptr;
}

/**
 * webkit_mime_info_unref:
 * @info: a #WebKitMimeInfo
 *
 * Atomically decrements the reference count of @info by one.
 *
 * If the reference count drops to 0, all memory allocated by the #WebKitMimeInfo is
 * released. This function is MT-safe and may be called from any
 * thread.
 *
 * Deprecated: 2.32
 */
void webkit_mime_info_unref(WebKitMimeInfo*)
{
}

/**
 * webkit_mime_info_get_mime_type:
 * @info: a #WebKitMimeInfo
 *
 * Gets the MIME type.
 *
 * Returns: MIME type, as a string.
 *
 * Deprecated: 2.32
 */
const char* webkit_mime_info_get_mime_type(WebKitMimeInfo*)
{
    return nullptr;
}

/**
 * webkit_mime_info_get_description:
 * @info: a #WebKitMimeInfo
 *
 * Gets the description of the MIME type.
 *
 * Returns: (nullable): description, as a string.
 *
 * Deprecated: 2.32
 */
const char* webkit_mime_info_get_description(WebKitMimeInfo*)
{
    return nullptr;
}

/**
 * webkit_mime_info_get_extensions:
 * @info: a #WebKitMimeInfo
 *
 * Get the list of file extensions associated to the MIME type.
 *
 * Returns: (array zero-terminated=1) (transfer none): a
 *     %NULL-terminated array of strings
 *
 * Deprecated: 2.32
 */
const char* const* webkit_mime_info_get_extensions(WebKitMimeInfo*)
{
    return nullptr;
}
