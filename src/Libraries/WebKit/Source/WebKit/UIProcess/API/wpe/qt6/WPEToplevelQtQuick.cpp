/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#include "WPEToplevelQtQuick.h"

#include <wtf/glib/WTFGType.h>

/**
 * WPEToplevelQtQuick:
 *
 */
struct _WPEToplevelQtQuickPrivate {
};
WEBKIT_DEFINE_FINAL_TYPE(WPEToplevelQtQuick, wpe_toplevel_qtquick, WPE_TYPE_TOPLEVEL, WPEToplevel)

static gboolean wpeToplevelQtQuickResize(WPEToplevel* toplevel, int width, int height)
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

static void wpe_toplevel_qtquick_class_init(WPEToplevelQtQuickClass* toplevelQtQuickClass)
{
    WPEToplevelClass* toplevelClass = WPE_TOPLEVEL_CLASS(toplevelQtQuickClass);
    toplevelClass->resize = wpeToplevelQtQuickResize;
}

WPEToplevel* wpe_toplevel_qtquick_new(WPEDisplayQtQuick* display)
{
    g_return_val_if_fail(WPE_IS_DISPLAY_QTQUICK(display), nullptr);
    return WPE_TOPLEVEL(g_object_new(WPE_TYPE_TOPLEVEL_QTQUICK, "display", display, nullptr));
}
