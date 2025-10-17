/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#ifndef WPEBufferSHM_h
#define WPEBufferSHM_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEBuffer.h>

G_BEGIN_DECLS

#define WPE_TYPE_BUFFER_SHM (wpe_buffer_shm_get_type())
WPE_API G_DECLARE_FINAL_TYPE (WPEBufferSHM, wpe_buffer_shm, WPE, BUFFER_SHM, WPEBuffer)

/**
 * WPEPixelFormat:
 * @WPE_PIXEL_FORMAT_ARGB8888: 32-bit ARGB format
 *
 * Enum with the supported pixel formats for memory buffers.
 */
typedef enum {
    WPE_PIXEL_FORMAT_ARGB8888
} WPEPixelFormat;

WPE_API WPEBufferSHM  *wpe_buffer_shm_new        (WPEView       *view,
                                                  int            width,
                                                  int            height,
                                                  WPEPixelFormat format,
                                                  GBytes        *data,
                                                  guint          stride);
WPE_API WPEPixelFormat wpe_buffer_shm_get_format (WPEBufferSHM  *buffer);
WPE_API GBytes        *wpe_buffer_shm_get_data   (WPEBufferSHM  *buffer);
WPE_API guint          wpe_buffer_shm_get_stride (WPEBufferSHM  *buffer);

G_END_DECLS

#endif /* WPEBuffer_h */
