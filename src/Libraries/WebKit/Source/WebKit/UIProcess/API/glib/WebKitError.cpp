/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
#include "WebKitError.h"

#include "APIError.h"
#include "WebKitPrivate.h"

using namespace WebCore;

/**
 * webkit_network_error_quark:
 *
 * Gets the quark for the domain of networking errors.
 *
 * Returns: network error domain.
 */
GQuark webkit_network_error_quark()
{
    return g_quark_from_static_string(reinterpret_cast<const char*>(API::Error::webKitNetworkErrorDomain().span8().data()));
}

/**
 * webkit_policy_error_quark:
 *
 * Gets the quark for the domain of policy errors.
 *
 * Returns: policy error domain.
 */
GQuark webkit_policy_error_quark()
{
    return g_quark_from_static_string(reinterpret_cast<const char*>(API::Error::webKitPolicyErrorDomain().span8().data()));
}

/**
 * webkit_plugin_error_quark:
 *
 * Gets the quark for the domain of plug-in errors.
 *
 * Returns: plug-in error domain.
 */
GQuark webkit_plugin_error_quark()
{
    return g_quark_from_static_string(reinterpret_cast<const char*>(API::Error::webKitPluginErrorDomain().span8().data()));
}

/**
 * webkit_download_error_quark:
 *
 * Gets the quark for the domain of download errors.
 *
 * Returns: download error domain.
 */
GQuark webkit_download_error_quark()
{
    return g_quark_from_static_string(reinterpret_cast<const char*>(API::Error::webKitDownloadErrorDomain().span8().data()));
}

#if PLATFORM(GTK)
/**
 * webkit_print_error_quark:
 *
 * Gets the quark for the domain of printing errors.
 *
 * Returns: print error domain.
 */
GQuark webkit_print_error_quark()
{
    return g_quark_from_static_string(reinterpret_cast<const char*>(API::Error::webKitPrintErrorDomain().span8().data()));
}
#endif

/**
 * webkit_javascript_error_quark:
 *
 * Gets the quark for the domain of JavaScript errors.
 *
 * Returns: JavaScript error domain.
 */
GQuark webkit_javascript_error_quark()
{
    return g_quark_from_static_string("WebKitJavascriptError");
}

/**
 * webkit_snapshot_error_quark:
 *
 * Gets the quark for the domain of page snapshot errors.
 *
 * Returns: snapshot error domain.
 */
GQuark webkit_snapshot_error_quark()
{
    return g_quark_from_static_string("WebKitSnapshotError");
}

/**
 * webkit_web_extension_match_pattern_error_quark:
 *
 * Gets the quark for the domain of Web Extension Match Pattern errors.
 *
 * Returns: web extension match pattern error domain.
 */
GQuark webkit_web_extension_match_pattern_error_quark()
{
    return g_quark_from_static_string("WebKitWebExtensionMatchPatternError");
}

/**
 * webkit_user_content_filter_error_quark:
 *
 * Gets the quark for the domain of user content filter errors.
 *
 * Returns: user content filter error domain.
 */
G_DEFINE_QUARK(WebKitUserContentFilterError, webkit_user_content_filter_error)

#if ENABLE(2022_GLIB_API)
/**
 * webkit_media_error_quark:
 *
 * Gets the quark for the domain of media errors.
 *
 * Returns: media error domin.
 *
 * Since: 2.40
 */
G_DEFINE_QUARK(WebKitMediaError, webkit_media_error)
#endif
