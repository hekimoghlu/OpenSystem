/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#include "WPEToplevelHeadless.h"

#include <wtf/glib/WTFGType.h>

/**
 * WPEToplevelHeadless:
 *
 */
struct _WPEToplevelHeadlessPrivate {
};
WEBKIT_DEFINE_FINAL_TYPE(WPEToplevelHeadless, wpe_toplevel_headless, WPE_TYPE_TOPLEVEL, WPEToplevel)

static void wpeToplevelHeadlessConstructed(GObject* object)
{
    G_OBJECT_CLASS(wpe_toplevel_headless_parent_class)->constructed(object);

    wpe_toplevel_state_changed(WPE_TOPLEVEL(object), WPE_TOPLEVEL_STATE_ACTIVE);
}

static gboolean wpeToplevelHeadlessResize(WPEToplevel* toplevel, int width, int height)
{
    wpe_toplevel_resized(toplevel, width, height);
    wpe_toplevel_foreach_view(toplevel, [](WPEToplevel* toplevel, WPEView* view, gpointer) -> gboolean {
        int width, height;
        wpe_toplevel_get_size(toplevel, &width, &height);
        wpe_view_resized(view, width, height);
        return FALSE;
    }, nullptr);
    return TRUE;
}

static gboolean wpeToplevelHeadlessSetFullscreen(WPEToplevel* toplevel, gboolean fullscreen)
{
    auto state = wpe_toplevel_get_state(toplevel);
    if (fullscreen)
        state = static_cast<WPEToplevelState>(state | WPE_TOPLEVEL_STATE_FULLSCREEN);
    else
        state = static_cast<WPEToplevelState>(state & ~WPE_TOPLEVEL_STATE_FULLSCREEN);
    wpe_toplevel_state_changed(toplevel, state);
    return TRUE;
}

static void wpe_toplevel_headless_class_init(WPEToplevelHeadlessClass* toplevelHeadlessClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(toplevelHeadlessClass);
    objectClass->constructed = wpeToplevelHeadlessConstructed;

    WPEToplevelClass* toplevelClass = WPE_TOPLEVEL_CLASS(toplevelHeadlessClass);
    toplevelClass->resize = wpeToplevelHeadlessResize;
    toplevelClass->set_fullscreen = wpeToplevelHeadlessSetFullscreen;
}

/**
 * wpe_toplevel_headless_new:
 * @display: a #WPEDisplayHeadless
 *
 * Create a new #WPEToplevel on @display.
 *
 * Returns: (transfer full): a #WPEToplevel
 */
WPEToplevel* wpe_toplevel_headless_new(WPEDisplayHeadless* display)
{
    return WPE_TOPLEVEL(g_object_new(WPE_TYPE_TOPLEVEL_HEADLESS, "display", display, nullptr));
}
