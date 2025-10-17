/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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

#if USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayerBuffer.h"

namespace WebCore {

class BitmapTexture;

class CoordinatedPlatformLayerBufferYUV final : public CoordinatedPlatformLayerBuffer {
public:
    enum class YuvToRgbColorSpace : uint8_t { BT601, BT709, BT2020, SMPTE240M };
    static std::unique_ptr<CoordinatedPlatformLayerBufferYUV> create(unsigned planeCount, std::array<unsigned, 4>&& planes, std::array<unsigned, 4>&& yuvPlane, std::array<unsigned, 4>&& yuvPlaneOffset, YuvToRgbColorSpace, const IntSize&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    static std::unique_ptr<CoordinatedPlatformLayerBufferYUV> create(unsigned planeCount, Vector<RefPtr<BitmapTexture>, 4>&& textures, std::array<unsigned, 4>&& yuvPlane, std::array<unsigned, 4>&& yuvPlaneOffset, YuvToRgbColorSpace, const IntSize&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    CoordinatedPlatformLayerBufferYUV(unsigned planeCount, std::array<unsigned, 4>&& planes, std::array<unsigned, 4>&& yuvPlane, std::array<unsigned, 4>&& yuvPlaneOffset, YuvToRgbColorSpace, const IntSize&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    CoordinatedPlatformLayerBufferYUV(unsigned planeCount, Vector<RefPtr<BitmapTexture>, 4>&& textures, std::array<unsigned, 4>&& yuvPlane, std::array<unsigned, 4>&& yuvPlaneOffset, YuvToRgbColorSpace, const IntSize&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    virtual ~CoordinatedPlatformLayerBufferYUV();

private:
    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) override;

    unsigned m_planeCount { 0 };
    Vector<RefPtr<BitmapTexture>, 4> m_textures;
    std::array<unsigned, 4> m_planes;
    std::array<unsigned, 4> m_yuvPlane;
    std::array<unsigned, 4> m_yuvPlaneOffset;
    YuvToRgbColorSpace m_yuvToRgbColorSpace { YuvToRgbColorSpace::BT601 };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(CoordinatedPlatformLayerBufferYUV, Type::YUV)

#endif // USE(COORDINATED_GRAPHICS)
