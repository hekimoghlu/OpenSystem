/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "WebKitPlugin.h"

#include <wtf/glib/WTFGType.h>

/**
 * WebKitPlugin:
 *
 * Represents a plugin, enabling fine-grained control.
 *
 * This object represents a single plugin, found while scanning the
 * various platform plugin directories. This object can be used to get
 * more information about a plugin, and enable/disable it, allowing
 * fine-grained control of plugins. The list of available plugins can
 * be obtained from the #WebKitWebContext, with
 * webkit_web_context_get_plugins().
 *
 * Deprecated: 2.32
 */

struct _WebKitPluginPrivate {
};

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
WEBKIT_DEFINE_TYPE(WebKitPlugin, webkit_plugin, G_TYPE_OBJECT)
ALLOW_DEPRECATED_DECLARATIONS_END

static void webkit_plugin_class_init(WebKitPluginClass*)
{
}

/**
 * webkit_plugin_get_name:
 * @plugin: a #WebKitPlugin
 *
 * Obtain the plugin name.
 *
 * Returns: (nullable): name, as a string.
 *
 * Deprecated: 2.32
 */
const char* webkit_plugin_get_name(WebKitPlugin*)
{
    return nullptr;
}

/**
 * webkit_plugin_get_description:
 * @plugin: a #WebKitPlugin
 *
 * Obtain the plugin description.
 *
 * Returns: (nullable): description, as a string.
 *
 * Deprecated: 2.32
 */
const char* webkit_plugin_get_description(WebKitPlugin*)
{
    return nullptr;
}

/**
 * webkit_plugin_get_path:
 * @plugin: a #WebKitPlugin
 *
 * Obtain the absolute path where the plugin is installed.
 *
 * Returns: (nullable): path, as a string.
 *
 * Deprecated: 2.32
 */
const char* webkit_plugin_get_path(WebKitPlugin*)
{
    return nullptr;
}

/**
 * webkit_plugin_get_mime_info_list:
 * @plugin: a #WebKitPlugin
 *
 * Get information about MIME types handled by the plugin.
 *
 * Get information about MIME types handled by the plugin,
 * as a list of #WebKitMimeInfo.
 *
 * Returns: (element-type WebKitMimeInfo) (transfer none): a #GList of #WebKitMimeInfo.
 *
 * Deprecated: 2.32
 */
GList* webkit_plugin_get_mime_info_list(WebKitPlugin*)
{
    return nullptr;
}
