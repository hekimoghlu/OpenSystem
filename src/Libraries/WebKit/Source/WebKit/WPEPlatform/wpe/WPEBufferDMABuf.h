/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#ifndef WPEBufferDMABuf_h
#define WPEBufferDMABuf_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEBuffer.h>

G_BEGIN_DECLS

#define WPE_TYPE_BUFFER_DMA_BUF (wpe_buffer_dma_buf_get_type())
WPE_API G_DECLARE_FINAL_TYPE (WPEBufferDMABuf, wpe_buffer_dma_buf, WPE, BUFFER_DMA_BUF, WPEBuffer)

WPE_API WPEBufferDMABuf *wpe_buffer_dma_buf_new          (WPEView         *view,
                                                          int              width,
                                                          int              height,
                                                          guint32          format,
                                                          guint32          n_planes,
                                                          int             *fds,
                                                          guint32         *offsets,
                                                          guint32         *strides,
                                                          guint64          modifier);
WPE_API guint32          wpe_buffer_dma_buf_get_format   (WPEBufferDMABuf *buffer);
WPE_API guint32          wpe_buffer_dma_buf_get_n_planes (WPEBufferDMABuf* buffer);
WPE_API int              wpe_buffer_dma_buf_get_fd       (WPEBufferDMABuf *buffer,
                                                          guint32          plane);
WPE_API guint32          wpe_buffer_dma_buf_get_offset   (WPEBufferDMABuf *buffer,
                                                          guint32          plane);
WPE_API guint32          wpe_buffer_dma_buf_get_stride   (WPEBufferDMABuf *buffer,
                                                          guint32          plane);
WPE_API guint64          wpe_buffer_dma_buf_get_modifier (WPEBufferDMABuf *buffer);
WPE_API void             wpe_buffer_dma_buf_set_rendering_fence  (WPEBufferDMABuf *buffer,
                                                                  int              fd);
WPE_API int              wpe_buffer_dma_buf_get_rendering_fence  (WPEBufferDMABuf *buffer);
WPE_API int              wpe_buffer_dma_buf_take_rendering_fence (WPEBufferDMABuf *buffer);
WPE_API void             wpe_buffer_dma_buf_set_release_fence    (WPEBufferDMABuf *buffer,
                                                                  int              fd);
WPE_API int              wpe_buffer_dma_buf_get_release_fence    (WPEBufferDMABuf *buffer);
WPE_API int              wpe_buffer_dma_buf_take_release_fence   (WPEBufferDMABuf *buffer);

G_END_DECLS

#endif /* WPEBuffer_h */
