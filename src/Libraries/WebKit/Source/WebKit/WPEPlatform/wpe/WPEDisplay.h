/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#ifndef WPEDisplay_h
#define WPEDisplay_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEBufferDMABufFormats.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEInputMethodContext.h>
#include <wpe/WPEKeymap.h>
#include <wpe/WPEScreen.h>
#include <wpe/WPESettings.h>
#include <wpe/WPEView.h>

G_BEGIN_DECLS

#define WPE_DISPLAY_EXTENSION_POINT_NAME "wpe-platform-display"

#define WPE_TYPE_DISPLAY (wpe_display_get_type())
WPE_DECLARE_DERIVABLE_TYPE (WPEDisplay, wpe_display, WPE, DISPLAY, GObject)

struct _WPEDisplayClass
{
    GObjectClass parent_class;

    gboolean                (* connect)                       (WPEDisplay *display,
                                                               GError    **error);
    WPEView                *(* create_view)                   (WPEDisplay *display);
    gpointer                (* get_egl_display)               (WPEDisplay *display,
                                                               GError    **error);
    WPEKeymap              *(* get_keymap)                    (WPEDisplay *display,
                                                               GError    **error);
    WPEBufferDMABufFormats *(* get_preferred_dma_buf_formats) (WPEDisplay *display);
    guint                   (* get_n_screens)                 (WPEDisplay *display);
    WPEScreen              *(* get_screen)                    (WPEDisplay *display,
                                                               guint       index);
    const char             *(* get_drm_device)                (WPEDisplay *display);
    const char             *(* get_drm_render_node)           (WPEDisplay *display);
    gboolean                (* use_explicit_sync)             (WPEDisplay *display);

    WPEInputMethodContext   *(* create_input_method_context)    (WPEDisplay *display);

    gpointer padding[32];
};

#define WPE_DISPLAY_ERROR (wpe_display_error_quark())

/**
 * WPEDisplayError:
 * @WPE_DISPLAY_ERROR_NOT_SUPPORTED: Operation not supported
 * @WPE_DISPLAY_ERROR_CONNECTION_FAILED: Failed to connect to the native system
 *
 * #WPEDisplay errors
 */
typedef enum {
    WPE_DISPLAY_ERROR_NOT_SUPPORTED,
    WPE_DISPLAY_ERROR_CONNECTION_FAILED
} WPEDisplayError;

WPE_API GQuark                  wpe_display_error_quark                   (void);
WPE_API WPEDisplay             *wpe_display_get_default                   (void);
WPE_API WPEDisplay             *wpe_display_get_primary                   (void);
WPE_API void                    wpe_display_set_primary                   (WPEDisplay *display);
WPE_API gboolean                wpe_display_connect                       (WPEDisplay *display,
                                                                           GError    **error);
WPE_API gpointer                wpe_display_get_egl_display               (WPEDisplay *display,
                                                                           GError    **error);
WPE_API WPEKeymap              *wpe_display_get_keymap                    (WPEDisplay *display,
                                                                           GError    **error);
WPE_API WPEBufferDMABufFormats *wpe_display_get_preferred_dma_buf_formats (WPEDisplay *display);
WPE_API guint                   wpe_display_get_n_screens                 (WPEDisplay *display);
WPE_API WPEScreen              *wpe_display_get_screen                    (WPEDisplay *display,
                                                                           guint       index);
WPE_API void                    wpe_display_screen_added                  (WPEDisplay *display,
                                                                           WPEScreen *screen);
WPE_API void                    wpe_display_screen_removed                (WPEDisplay *display,
                                                                           WPEScreen *screen);
WPE_API const char             *wpe_display_get_drm_device                (WPEDisplay *display);
WPE_API const char             *wpe_display_get_drm_render_node           (WPEDisplay *display);
WPE_API gboolean                wpe_display_use_explicit_sync             (WPEDisplay *display);

WPE_API WPESettings            *wpe_display_get_settings                  (WPEDisplay *display);

G_END_DECLS

#endif /* WPEDisplay_h */
