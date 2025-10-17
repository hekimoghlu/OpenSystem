/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "WPEExtensions.h"

#include "WPEDisplay.h"
#include <gio/gio.h>
#include <mutex>
#include <wtf/glib/GUniquePtr.h>

#if ENABLE(WPE_PLATFORM_DRM)
#include "wpe/drm/WPEDisplayDRM.h"
#endif

#if ENABLE(WPE_PLATFORM_HEADLESS)
#include "wpe/headless/WPEDisplayHeadless.h"
#endif

#if ENABLE(WPE_PLATFORM_WAYLAND)
#include "wpe/wayland/WPEDisplayWayland.h"
#endif

void wpeEnsureExtensionPointsRegistered()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        auto* extensionPoint = g_io_extension_point_register(WPE_DISPLAY_EXTENSION_POINT_NAME);
        g_io_extension_point_set_required_type(extensionPoint, WPE_TYPE_DISPLAY);
    });
}

void wpeEnsureExtensionPointsLoaded()
{
    wpeEnsureExtensionPointsRegistered();

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        GIOModuleScope* scope = g_io_module_scope_new(G_IO_MODULE_SCOPE_BLOCK_DUPLICATES);
        const char* path = g_getenv("WPE_PLATFORMS_PATH");
        if (path && *path) {
            GUniquePtr<char*> paths(g_strsplit(path, G_SEARCHPATH_SEPARATOR_S, 0));
            for (size_t i = 0; paths.get()[i]; ++i)
                g_io_modules_scan_all_in_directory_with_scope(paths.get()[i], scope);
        }

        g_io_modules_scan_all_in_directory_with_scope(WPE_PLATFORM_MODULE_DIR, scope);
        g_io_module_scope_free(scope);

        // Initialize types of builtin extensions.
#if ENABLE(WPE_PLATFORM_DRM)
        g_type_ensure(wpe_display_drm_get_type());
#endif
#if ENABLE(WPE_PLATFORM_HEADLESS)
        g_type_ensure(wpe_display_headless_get_type());
#endif
#if ENABLE(WPE_PLATFORM_WAYLAND)
        g_type_ensure(wpe_display_wayland_get_type());
#endif
    });
}
