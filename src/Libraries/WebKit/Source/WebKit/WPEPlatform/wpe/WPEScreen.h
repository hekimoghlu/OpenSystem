/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#ifndef WPEScreen_h
#define WPEScreen_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>

G_BEGIN_DECLS

#define WPE_TYPE_SCREEN (wpe_screen_get_type())
WPE_DECLARE_DERIVABLE_TYPE (WPEScreen, wpe_screen, WPE, SCREEN, GObject)

struct _WPEScreenClass
{
    GObjectClass parent_class;

    void (* invalidate) (WPEScreen *screen);

    gpointer padding[32];
};

WPE_API guint32 wpe_screen_get_id               (WPEScreen *screen);
WPE_API void    wpe_screen_invalidate           (WPEScreen *screen);
WPE_API int     wpe_screen_get_x                (WPEScreen *screen);
WPE_API int     wpe_screen_get_y                (WPEScreen *screen);
WPE_API void    wpe_screen_set_position         (WPEScreen *screen,
                                                 int         x,
                                                 int         y);
WPE_API int     wpe_screen_get_width            (WPEScreen *screen);
WPE_API int     wpe_screen_get_height           (WPEScreen *screen);
WPE_API void    wpe_screen_set_size             (WPEScreen *screen,
                                                 int         width,
                                                 int         height);
WPE_API int     wpe_screen_get_physical_width   (WPEScreen *screen);
WPE_API int     wpe_screen_get_physical_height  (WPEScreen *screen);
WPE_API void    wpe_screen_set_physical_size    (WPEScreen *screen,
                                                 int         width,
                                                 int         height);
WPE_API gdouble wpe_screen_get_scale            (WPEScreen *screen);
WPE_API void    wpe_screen_set_scale            (WPEScreen *screen,
                                                 gdouble     scale);
WPE_API int     wpe_screen_get_refresh_rate     (WPEScreen *screen);
WPE_API void    wpe_screen_set_refresh_rate     (WPEScreen *screen,
                                                 int         refresh_rate);

G_END_DECLS

#endif /* WPEScreen_h */
