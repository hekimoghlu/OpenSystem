/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#include "CoordinatedPlatformLayerBufferExternalOES.h"

#if USE(COORDINATED_GRAPHICS)
#include "TextureMapper.h"

namespace WebCore {

std::unique_ptr<CoordinatedPlatformLayerBufferExternalOES> CoordinatedPlatformLayerBufferExternalOES::create(unsigned textureID, const IntSize& size, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
{
    return makeUnique<CoordinatedPlatformLayerBufferExternalOES>(textureID, size, flags, WTFMove(fence));
}

CoordinatedPlatformLayerBufferExternalOES::CoordinatedPlatformLayerBufferExternalOES(unsigned textureID, const IntSize& size, OptionSet<TextureMapperFlags> flags, std::unique_ptr<GLFence>&& fence)
    : CoordinatedPlatformLayerBuffer(Type::ExternalOES, size, flags, WTFMove(fence))
    , m_textureID(textureID)
{
}

CoordinatedPlatformLayerBufferExternalOES::~CoordinatedPlatformLayerBufferExternalOES() = default;

void CoordinatedPlatformLayerBufferExternalOES::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& modelViewMatrix, float opacity)
{
    waitForContentsIfNeeded();
    textureMapper.drawTextureExternalOES(m_textureID, m_flags, targetRect, modelViewMatrix, opacity);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
