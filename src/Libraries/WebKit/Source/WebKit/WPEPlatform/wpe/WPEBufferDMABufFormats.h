/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#ifndef WPEBufferDMABufFormats_h
#define WPEBufferDMABufFormats_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>

G_BEGIN_DECLS

#define WPE_TYPE_BUFFER_DMA_BUF_FORMATS (wpe_buffer_dma_buf_formats_get_type())
WPE_API G_DECLARE_FINAL_TYPE (WPEBufferDMABufFormats, wpe_buffer_dma_buf_formats, WPE, BUFFER_DMA_BUF_FORMATS, GObject)

/**
 * WPEBufferDMABufFormatUsage:
 * @WPE_BUFFER_DMA_BUF_FORMAT_USAGE_RENDERING: format should be used for rendering.
 * @WPE_BUFFER_DMA_BUF_FORMAT_USAGE_MAPPING: format should be used for mapping buffer.
 * @WPE_BUFFER_DMA_BUF_FORMAT_USAGE_SCANOUT: format should be used for scanout.
 *
 * Enum values to indicate the best usage of a #WPEBufferDMABufFormat.
 */
typedef enum {
    WPE_BUFFER_DMA_BUF_FORMAT_USAGE_RENDERING,
    WPE_BUFFER_DMA_BUF_FORMAT_USAGE_MAPPING,
    WPE_BUFFER_DMA_BUF_FORMAT_USAGE_SCANOUT
} WPEBufferDMABufFormatUsage;

WPE_API const char                *wpe_buffer_dma_buf_formats_get_device           (WPEBufferDMABufFormats *formats);
WPE_API guint                      wpe_buffer_dma_buf_formats_get_n_groups         (WPEBufferDMABufFormats *formats);
WPE_API WPEBufferDMABufFormatUsage wpe_buffer_dma_buf_formats_get_group_usage      (WPEBufferDMABufFormats *formats,
                                                                                    guint                   group);
WPE_API const char                *wpe_buffer_dma_buf_formats_get_group_device     (WPEBufferDMABufFormats *formats,
                                                                                    guint                   group);
WPE_API guint                      wpe_buffer_dma_buf_formats_get_group_n_formats  (WPEBufferDMABufFormats *formats,
                                                                                    guint                   group);
WPE_API guint32                    wpe_buffer_dma_buf_formats_get_format_fourcc    (WPEBufferDMABufFormats *formats,
                                                                                    guint                   group,
                                                                                    guint                   format);
WPE_API GArray                    *wpe_buffer_dma_buf_formats_get_format_modifiers (WPEBufferDMABufFormats *formats,
                                                                                    guint                   group,
                                                                                    guint                   format);

#define WPE_TYPE_BUFFER_DMA_BUF_FORMATS_BUILDER (wpe_buffer_dma_buf_formats_builder_get_type())
typedef struct _WPEBufferDMABufFormatsBuilder WPEBufferDMABufFormatsBuilder;

WPE_API GType                          wpe_buffer_dma_buf_formats_builder_get_type      (void);
WPE_API WPEBufferDMABufFormatsBuilder *wpe_buffer_dma_buf_formats_builder_new           (const char *device);
WPE_API WPEBufferDMABufFormatsBuilder *wpe_buffer_dma_buf_formats_builder_ref           (WPEBufferDMABufFormatsBuilder *builder);
WPE_API void                           wpe_buffer_dma_buf_formats_builder_unref         (WPEBufferDMABufFormatsBuilder *builder);
WPE_API void                           wpe_buffer_dma_buf_formats_builder_append_group  (WPEBufferDMABufFormatsBuilder *builder,
                                                                                         const char                    *device,
                                                                                         WPEBufferDMABufFormatUsage     usage);
WPE_API void                           wpe_buffer_dma_buf_formats_builder_append_format (WPEBufferDMABufFormatsBuilder *builder,
                                                                                         guint32                        fourcc,
                                                                                         guint64                        modifier);
WPE_API WPEBufferDMABufFormats        *wpe_buffer_dma_buf_formats_builder_end           (WPEBufferDMABufFormatsBuilder *builder);

G_END_DECLS

#endif /* WPEBufferDMABufFormats_h */
