/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#include "WebKitInstallMissingMediaPluginsPermissionRequest.h"

#if !ENABLE(2022_GLIB_API)

#include "WebKitPermissionRequest.h"
#include <wtf/glib/WTFGType.h>

/**
 * WebKitInstallMissingMediaPluginsPermissionRequest:
 * @See_also: #WebKitPermissionRequest, #WebKitWebView
 *
 * Previously, a permission request for installing missing media plugins.
 *
 * WebKitInstallMissingMediaPluginsPermissionRequest will no longer ever be created, so
 * you can remove any code that attempts to handle it.
 *
 * Since: 2.10
 *
 * Deprecated: 2.40
 */

static void webkit_permission_request_interface_init(WebKitPermissionRequestIface*);

struct _WebKitInstallMissingMediaPluginsPermissionRequestPrivate {
};

WEBKIT_DEFINE_TYPE_WITH_CODE(
    WebKitInstallMissingMediaPluginsPermissionRequest, webkit_install_missing_media_plugins_permission_request, G_TYPE_OBJECT,
    G_IMPLEMENT_INTERFACE(WEBKIT_TYPE_PERMISSION_REQUEST, webkit_permission_request_interface_init))

static void webkit_permission_request_interface_init(WebKitPermissionRequestIface* iface)
{
    iface->allow = [](auto*) { };
    iface->deny = [](auto*) { };
}

static void webkit_install_missing_media_plugins_permission_request_class_init(WebKitInstallMissingMediaPluginsPermissionRequestClass*)
{
}

/**
 * webkit_install_missing_media_plugins_permission_request_get_description:
 * @request: a #WebKitInstallMissingMediaPluginsPermissionRequest
 *
 * This function returns an empty string.
 *
 * Returns: an empty string
 *
 * Since: 2.10
 *
 * Deprecated: 2.40
 */
const char* webkit_install_missing_media_plugins_permission_request_get_description(WebKitInstallMissingMediaPluginsPermissionRequest*)
{
    return "";
}

#endif // !ENABLE(2022_GLIB_API)
