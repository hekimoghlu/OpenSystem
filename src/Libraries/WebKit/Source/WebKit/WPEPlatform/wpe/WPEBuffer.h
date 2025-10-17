/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#ifndef WPEBuffer_h
#define WPEBuffer_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEView.h>

G_BEGIN_DECLS

#define WPE_TYPE_BUFFER (wpe_buffer_get_type())
WPE_DECLARE_DERIVABLE_TYPE (WPEBuffer, wpe_buffer, WPE, BUFFER, GObject)

struct _WPEBufferClass
{
    GObjectClass parent_class;

    gpointer (* import_to_egl_image) (WPEBuffer *buffer,
                                      GError   **error);
    GBytes  *(* import_to_pixels)    (WPEBuffer *buffer,
                                      GError   **error);

    gpointer padding[32];
};

#define WPE_BUFFER_ERROR (wpe_buffer_error_quark())

/**
 * WPEBufferError:
 * @WPE_BUFFER_ERROR_NOT_SUPPORTED: Operation not supported
 * @WPE_BUFFER_ERROR_IMPORT_FAILED: Import buffer operation failed
 *
 * #WPEBuffer errors
 */
typedef enum {
    WPE_BUFFER_ERROR_NOT_SUPPORTED,
    WPE_BUFFER_ERROR_IMPORT_FAILED
} WPEBufferError;

WPE_API GQuark      wpe_buffer_error_quark         (void);
WPE_API WPEView    *wpe_buffer_get_view            (WPEBuffer     *buffer);
WPE_API int         wpe_buffer_get_width           (WPEBuffer     *buffer);
WPE_API int         wpe_buffer_get_height          (WPEBuffer     *buffer);
WPE_API void        wpe_buffer_set_user_data       (WPEBuffer     *buffer,
                                                    gpointer       user_data,
                                                    GDestroyNotify destroy_func);
WPE_API gpointer    wpe_buffer_get_user_data       (WPEBuffer     *buffer);
WPE_API gpointer    wpe_buffer_import_to_egl_image (WPEBuffer     *buffer,
                                                    GError       **error);
WPE_API GBytes     *wpe_buffer_import_to_pixels    (WPEBuffer     *buffer,
                                                    GError       **error);

G_END_DECLS

#endif /* WPEBuffer_h */
