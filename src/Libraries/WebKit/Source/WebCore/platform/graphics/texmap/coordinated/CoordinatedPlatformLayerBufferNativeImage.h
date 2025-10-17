/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

class NativeImage;

class CoordinatedPlatformLayerBufferNativeImage final : public CoordinatedPlatformLayerBuffer {
public:
    static std::unique_ptr<CoordinatedPlatformLayerBufferNativeImage> create(Ref<NativeImage>&&, std::unique_ptr<GLFence>&&);
    CoordinatedPlatformLayerBufferNativeImage(Ref<NativeImage>&&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    virtual ~CoordinatedPlatformLayerBufferNativeImage();

    const NativeImage& image() const { return m_image.get(); }

private:
    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) override;

    bool tryEnsureBuffer(TextureMapper&);

    Ref<NativeImage> m_image;
    std::unique_ptr<CoordinatedPlatformLayerBuffer> m_buffer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(CoordinatedPlatformLayerBufferNativeImage, Type::NativeImage)

#endif // USE(COORDINATED_GRAPHICS)
