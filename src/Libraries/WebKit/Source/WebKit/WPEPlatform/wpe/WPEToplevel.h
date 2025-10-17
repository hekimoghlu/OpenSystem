/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
#ifndef WPEToplevel_h
#define WPEToplevel_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>

G_BEGIN_DECLS

#define WPE_TYPE_TOPLEVEL (wpe_toplevel_get_type())
WPE_DECLARE_DERIVABLE_TYPE (WPEToplevel, wpe_toplevel, WPE, TOPLEVEL, GObject)

typedef struct _WPEBufferDMABufFormats WPEBufferDMABufFormats;
typedef struct _WPEDisplay WPEDisplay;
typedef struct _WPEScreen WPEScreen;
typedef struct _WPEView WPEView;

struct _WPEToplevelClass
{
    GObjectClass parent_class;

    void                    (* set_title)                     (WPEToplevel *toplevel,
                                                               const char  *title);
    guint                   (* get_max_views)                 (WPEToplevel *toplevel);
    WPEScreen              *(* get_screen)                    (WPEToplevel *toplevel);
    gboolean                (* resize)                        (WPEToplevel *toplevel,
                                                               int          width,
                                                               int          height);
    gboolean                (* set_fullscreen)                (WPEToplevel *toplevel,
                                                               gboolean     fullscreen);
    gboolean                (* set_maximized)                 (WPEToplevel *toplevel,
                                                               gboolean     maximized);
    gboolean                (* set_minimized)                 (WPEToplevel *toplevel);
    WPEBufferDMABufFormats *(* get_preferred_dma_buf_formats) (WPEToplevel *toplevel);

    gpointer padding[32];
};

/**
 * WPEToplevelState:
 * @WPE_TOPLEVEL_STATE_NONE: the toplevel is in normal state
 * @WPE_TOPLEVEL_STATE_FULLSCREEN: the toplevel is fullscreen
 * @WPE_TOPLEVEL_STATE_MAXIMIZED: the toplevel is maximized
 * @WPE_TOPLEVEL_STATE_ACTIVE: the toplevel is active
 *
 * The current state of a #WPEToplevel.
 */
typedef enum {
    WPE_TOPLEVEL_STATE_NONE       = 0,
    WPE_TOPLEVEL_STATE_FULLSCREEN = (1 << 0),
    WPE_TOPLEVEL_STATE_MAXIMIZED  = (1 << 1),
    WPE_TOPLEVEL_STATE_ACTIVE     = (1 << 2)
} WPEToplevelState;

/**
 * WPEToplevelForeachViewFunc:
 * @toplevel: a #WPEToplevel
 * @view: a #WPEView
 * @user_data: (closure): user data
 *
 * A function used by wpe_toplevel_foreach_view().
 *
 * Returns: %TRUE to stop the iteration, or %FALSE to continue
 */
typedef gboolean (* WPEToplevelForeachViewFunc) (WPEToplevel *toplevel,
                                                 WPEView     *view,
                                                 gpointer     user_data);

WPE_API WPEDisplay             *wpe_toplevel_get_display                       (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_set_title                         (WPEToplevel               *toplevel,
                                                                                const char                *title);
WPE_API guint                   wpe_toplevel_get_max_views                     (WPEToplevel               *toplevel);
WPE_API guint                   wpe_toplevel_get_n_views                       (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_foreach_view                      (WPEToplevel               *toplevel,
                                                                                WPEToplevelForeachViewFunc func,
                                                                                gpointer                   user_data);
WPE_API void                    wpe_toplevel_closed                            (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_get_size                          (WPEToplevel               *toplevel,
                                                                                int                       *width,
                                                                                int                       *height);
WPE_API gboolean                wpe_toplevel_resize                            (WPEToplevel               *toplevel,
                                                                                int                        width,
                                                                                int                        height);
WPE_API void                    wpe_toplevel_resized                           (WPEToplevel               *toplevel,
                                                                                int                        width,
                                                                                int                        height);
WPE_API WPEToplevelState        wpe_toplevel_get_state                         (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_state_changed                     (WPEToplevel               *toplevel,
                                                                                WPEToplevelState           state);
WPE_API gdouble                 wpe_toplevel_get_scale                         (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_scale_changed                     (WPEToplevel               *toplevel,
                                                                                gdouble                    scale);
WPE_API WPEScreen              *wpe_toplevel_get_screen                        (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_screen_changed                    (WPEToplevel               *toplevel);
WPE_API gboolean                wpe_toplevel_fullscreen                        (WPEToplevel               *toplevel);
WPE_API gboolean                wpe_toplevel_unfullscreen                      (WPEToplevel               *toplevel);
WPE_API gboolean                wpe_toplevel_maximize                          (WPEToplevel               *toplevel);
WPE_API gboolean                wpe_toplevel_unmaximize                        (WPEToplevel               *toplevel);
WPE_API gboolean                wpe_toplevel_minimize                          (WPEToplevel               *toplevel);
WPE_API WPEBufferDMABufFormats *wpe_toplevel_get_preferred_dma_buf_formats     (WPEToplevel               *toplevel);
WPE_API void                    wpe_toplevel_preferred_dma_buf_formats_changed (WPEToplevel               *toplevel);

G_END_DECLS

#endif /* WPEToplevel_h */
