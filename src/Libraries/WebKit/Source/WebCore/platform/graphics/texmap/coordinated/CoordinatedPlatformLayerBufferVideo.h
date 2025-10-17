/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
#pragma once

#if USE(COORDINATED_GRAPHICS) && ENABLE(VIDEO) && USE(GSTREAMER)
#include "CoordinatedPlatformLayerBuffer.h"
#include "GStreamerCommon.h"

namespace WebCore {

class CoordinatedPlatformLayerBufferVideo final : public CoordinatedPlatformLayerBuffer {
public:
    static std::unique_ptr<CoordinatedPlatformLayerBufferVideo> create(GstSample*, std::optional<GstVideoDecoderPlatform>, bool gstGLEnabled, OptionSet<TextureMapperFlags>);
    CoordinatedPlatformLayerBufferVideo(GstBuffer*, GstVideoInfo*, std::optional<std::pair<uint32_t, uint64_t>>, std::optional<GstVideoDecoderPlatform>, bool gstGLEnabled, OptionSet<TextureMapperFlags>);
    virtual ~CoordinatedPlatformLayerBufferVideo();

    std::unique_ptr<CoordinatedPlatformLayerBuffer> copyBuffer() const;

private:
    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) override;

    std::unique_ptr<CoordinatedPlatformLayerBuffer> createBufferIfNeeded(GstBuffer*, GstVideoInfo*, std::optional<std::pair<uint32_t, uint64_t>>, bool gstGLEnabled);
#if USE(GBM)
    std::unique_ptr<CoordinatedPlatformLayerBuffer> createBufferFromDMABufMemory(GstBuffer*, GstVideoInfo*, std::optional<std::pair<uint32_t, uint64_t>>);
#endif
    std::unique_ptr<CoordinatedPlatformLayerBuffer> createBufferFromGLMemory(GstBuffer*, GstVideoInfo*);

    GstVideoFrame m_videoFrame;
    std::optional<GstVideoDecoderPlatform> m_videoDecoderPlatform;
    bool m_isMapped { false };
    std::unique_ptr<CoordinatedPlatformLayerBuffer> m_buffer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(CoordinatedPlatformLayerBufferVideo, Type::Video)

#endif // USE(COORDINATED_GRAPHICS) && ENABLE(VIDEO) && USE(GSTREAMER)
