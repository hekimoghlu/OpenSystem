/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#include "config.h"
#include "CoordinatedPlatformLayerBufferRGB.h"

#if USE(COORDINATED_GRAPHICS)
#include "BitmapTexture.h"
#include "TextureMapper.h"

namespace WebCore {

std::unique_ptr<CoordinatedPlatformLayerBufferRGB> CoordinatedPlatformLayerBufferRGB::create(Ref<BitmapTexture>&& texture, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
{
    return makeUnique<CoordinatedPlatformLayerBufferRGB>(WTFMove(texture), flags, WTFMove(fence));
}

std::unique_ptr<CoordinatedPlatformLayerBufferRGB> CoordinatedPlatformLayerBufferRGB::create(unsigned textureID, const IntSize& size, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
{
    return makeUnique<CoordinatedPlatformLayerBufferRGB>(textureID, size, flags, WTFMove(fence));
}

CoordinatedPlatformLayerBufferRGB::CoordinatedPlatformLayerBufferRGB(Ref<BitmapTexture>&& texture, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
    : CoordinatedPlatformLayerBuffer(Type::RGB, texture->size(), flags, WTFMove(fence))
    , m_texture(WTFMove(texture))
{
}

CoordinatedPlatformLayerBufferRGB::CoordinatedPlatformLayerBufferRGB(unsigned textureID, const IntSize& size, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
    : CoordinatedPlatformLayerBuffer(Type::RGB, size, flags, WTFMove(fence))
    , m_textureID(textureID)
{
}

CoordinatedPlatformLayerBufferRGB::~CoordinatedPlatformLayerBufferRGB() = default;

void CoordinatedPlatformLayerBufferRGB::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& modelViewMatrix, float opacity)
{
    waitForContentsIfNeeded();

    if (m_texture)
        textureMapper.drawTexture(m_texture->id(), m_flags | m_texture->colorConvertFlags(), targetRect, modelViewMatrix, opacity);
    else
        textureMapper.drawTexture(m_textureID, m_flags, targetRect, modelViewMatrix, opacity);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
