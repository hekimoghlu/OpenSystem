/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#if USE(COORDINATED_GRAPHICS) && ENABLE(VIDEO)
#include "CoordinatedPlatformLayerBuffer.h"

#if USE(GSTREAMER)
#include "GStreamerCommon.h"
#endif

namespace WebCore {

class GStreamerQuirksManager;

class CoordinatedPlatformLayerBufferHolePunch final : public CoordinatedPlatformLayerBuffer {
public:
    static std::unique_ptr<CoordinatedPlatformLayerBufferHolePunch> create(const IntSize&);
    explicit CoordinatedPlatformLayerBufferHolePunch(const IntSize&);
#if USE(GSTREAMER)
    static std::unique_ptr<CoordinatedPlatformLayerBufferHolePunch> create(const IntSize&, GstElement*, RefPtr<GStreamerQuirksManager>&&);
    CoordinatedPlatformLayerBufferHolePunch(const IntSize&, GstElement*, RefPtr<GStreamerQuirksManager>&&);
#endif
    virtual ~CoordinatedPlatformLayerBufferHolePunch();

private:
    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) override;

    void notifyVideoPosition(const FloatRect&, const TransformationMatrix&) override;
    void paintTransparentRectangle(TextureMapper&, const FloatRect&, const TransformationMatrix&) override;

#if USE(GSTREAMER)
    GRefPtr<GstElement> m_videoSink;
    RefPtr<GStreamerQuirksManager> m_quirksManager;
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(CoordinatedPlatformLayerBufferHolePunch, Type::HolePunch)

#endif // USE(COORDINATED_GRAPHICS) && ENABLE(VIDEO)
