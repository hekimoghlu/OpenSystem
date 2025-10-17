/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
#include "WPEToplevelDRM.h"

#include "WPEDisplayDRMPrivate.h"
#include "WPEScreenDRMPrivate.h"
#include <wtf/glib/WTFGType.h>

/**
 * WPEToplevelDRM:
 *
 */
struct _WPEToplevelDRMPrivate {
};
WEBKIT_DEFINE_FINAL_TYPE(WPEToplevelDRM, wpe_toplevel_drm, WPE_TYPE_TOPLEVEL, WPEToplevel)

static void wpeToplevelDRMConstructed(GObject* object)
{
    G_OBJECT_CLASS(wpe_toplevel_drm_parent_class)->constructed(object);

    auto* toplevel = WPE_TOPLEVEL(object);
    auto* display = WPE_DISPLAY_DRM(wpe_toplevel_get_display(toplevel));
    auto* screen = wpeDisplayDRMGetScreen(display);
    auto* mode = wpeScreenDRMGetMode(WPE_SCREEN_DRM(screen));
    double scale = wpe_screen_get_scale(screen);
    wpe_toplevel_resized(toplevel, mode->hdisplay / scale, mode->vdisplay / scale);
    wpe_toplevel_scale_changed(toplevel, scale);
    wpe_toplevel_state_changed(toplevel, static_cast<WPEToplevelState>(WPE_TOPLEVEL_STATE_FULLSCREEN | WPE_TOPLEVEL_STATE_ACTIVE));
}

static WPEScreen* wpeToplevelDRMGetScreen(WPEToplevel* toplevel)
{
    if (auto* display = wpe_toplevel_get_display(toplevel))
        return wpeDisplayDRMGetScreen(WPE_DISPLAY_DRM(display));
    return nullptr;
}

static void wpe_toplevel_drm_class_init(WPEToplevelDRMClass* toplevelDRMClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(toplevelDRMClass);
    objectClass->constructed = wpeToplevelDRMConstructed;

    WPEToplevelClass* toplevelClass = WPE_TOPLEVEL_CLASS(toplevelDRMClass);
    toplevelClass->get_screen = wpeToplevelDRMGetScreen;
}

/**
 * wpe_toplevel_drm_new:
 * @display: a #WPEDisplayDRM
 *
 * Create a new #WPEToplevel on @display.
 *
 * Returns: (transfer full): a #WPEToplevel
 */
WPEToplevel* wpe_toplevel_drm_new(WPEDisplayDRM* display)
{
    return WPE_TOPLEVEL(g_object_new(WPE_TYPE_TOPLEVEL_DRM, "display", display, nullptr));
}
