/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#ifndef WPEView_h
#define WPEView_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEGestureController.h>
#include <wpe/WPEToplevel.h>

G_BEGIN_DECLS

#define WPE_TYPE_VIEW (wpe_view_get_type())
WPE_DECLARE_DERIVABLE_TYPE (WPEView, wpe_view, WPE, VIEW, GObject)

typedef struct _WPEBuffer WPEBuffer;
typedef struct _WPEBufferDMABufFormats WPEBufferDMABufFormats;
typedef struct _WPEDisplay WPEDisplay;
typedef struct _WPEEvent WPEEvent;
typedef struct _WPEScreen WPEScreen;
typedef struct _WPERectangle WPERectangle;

struct _WPEViewClass
{
    GObjectClass parent_class;

    gboolean (* render_buffer)         (WPEView            *view,
                                        WPEBuffer          *buffer,
                                        const WPERectangle *damage_rects,
                                        guint               n_damage_rects,
                                        GError            **error);
    gboolean (* lock_pointer)          (WPEView            *view);
    gboolean (* unlock_pointer)        (WPEView            *view);
    void     (* set_cursor_from_name)  (WPEView            *view,
                                        const char         *name);
    void     (* set_cursor_from_bytes) (WPEView            *view,
                                        GBytes             *bytes,
                                        guint               width,
                                        guint               height,
                                        guint               stride,
                                        guint               hotspot_x,
                                        guint               hotspot_y);
    void     (* set_opaque_rectangles) (WPEView            *view,
                                        WPERectangle       *rects,
                                        guint               n_rects);
    gboolean (* can_be_mapped)         (WPEView            *view);

    gpointer padding[32];
};

#define WPE_VIEW_ERROR (wpe_view_error_quark())

/**
 * WPEViewError:
 * @WPE_VIEW_ERROR_RENDER_FAILED: Failed to render
 *
 * #WPEView errors
 */
typedef enum {
    WPE_VIEW_ERROR_RENDER_FAILED
} WPEViewError;

WPE_API GQuark                  wpe_view_error_quark                   (void);

WPE_API WPEView                *wpe_view_new                           (WPEDisplay         *display);
WPE_API WPEDisplay             *wpe_view_get_display                   (WPEView            *view);
WPE_API WPEToplevel            *wpe_view_get_toplevel                  (WPEView            *view);
WPE_API void                    wpe_view_set_toplevel                  (WPEView            *view,
                                                                        WPEToplevel        *toplevel);
WPE_API int                     wpe_view_get_width                     (WPEView            *view);
WPE_API int                     wpe_view_get_height                    (WPEView            *view);
WPE_API void                    wpe_view_closed                        (WPEView            *view);
WPE_API void                    wpe_view_resized                       (WPEView            *view,
                                                                        int                 width,
                                                                        int                 height);
WPE_API gdouble                 wpe_view_get_scale                     (WPEView            *view);
WPE_API gboolean                wpe_view_get_visible                   (WPEView            *view);
WPE_API void                    wpe_view_set_visible                   (WPEView            *view,
                                                                        gboolean            visible);
WPE_API gboolean                wpe_view_get_mapped                    (WPEView            *view);
WPE_API void                    wpe_view_map                           (WPEView            *view);
WPE_API void                    wpe_view_unmap                         (WPEView            *view);
WPE_API gboolean                wpe_view_lock_pointer                  (WPEView            *view);
WPE_API gboolean                wpe_view_unlock_pointer                (WPEView            *view);
WPE_API void                    wpe_view_set_cursor_from_name          (WPEView            *view,
                                                                        const char         *name);
WPE_API void                    wpe_view_set_cursor_from_bytes         (WPEView            *view,
                                                                        GBytes             *bytes,
                                                                        guint               width,
                                                                        guint               height,
                                                                        guint               stride,
                                                                        guint               hotspot_x,
                                                                        guint               hotspot_y);
WPE_API WPEToplevelState        wpe_view_get_toplevel_state            (WPEView            *view);
WPE_API WPEScreen              *wpe_view_get_screen                    (WPEView            *view);
WPE_API gboolean                wpe_view_render_buffer                 (WPEView            *view,
                                                                        WPEBuffer          *buffer,
                                                                        const WPERectangle *damage_rects,
                                                                        guint               n_damage_rects,
                                                                        GError            **error);
WPE_API void                    wpe_view_buffer_rendered               (WPEView            *view,
                                                                        WPEBuffer          *buffer);
WPE_API void                    wpe_view_buffer_released               (WPEView            *view,
                                                                        WPEBuffer          *buffer);
WPE_API void                    wpe_view_event                         (WPEView            *view,
                                                                        WPEEvent           *event);
WPE_API guint                   wpe_view_compute_press_count           (WPEView            *view,
                                                                        gdouble             x,
                                                                        gdouble             y,
                                                                        guint               button,
                                                                        guint32             time);
WPE_API void                    wpe_view_focus_in                      (WPEView            *view);
WPE_API void                    wpe_view_focus_out                     (WPEView            *view);
WPE_API gboolean                wpe_view_get_has_focus                 (WPEView            *view);
WPE_API WPEBufferDMABufFormats *wpe_view_get_preferred_dma_buf_formats (WPEView            *view);
WPE_API void                    wpe_view_set_opaque_rectangles         (WPEView            *view,
                                                                        WPERectangle       *rects,
                                                                        guint               n_rects);
WPE_API void                    wpe_view_set_gesture_controller        (WPEView            *view,
                                                                        WPEGestureController *controller);
WPE_API WPEGestureController   *wpe_view_get_gesture_controller        (WPEView            *view);

G_END_DECLS

#endif /* WPEView_h */
