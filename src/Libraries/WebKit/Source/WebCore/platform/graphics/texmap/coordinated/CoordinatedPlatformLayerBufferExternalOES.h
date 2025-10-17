/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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

class CoordinatedPlatformLayerBufferExternalOES final : public CoordinatedPlatformLayerBuffer {
public:
    static std::unique_ptr<CoordinatedPlatformLayerBufferExternalOES> create(unsigned textureID, const IntSize&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    CoordinatedPlatformLayerBufferExternalOES(unsigned textureID, const IntSize&, OptionSet<TextureMapperFlags>, std::unique_ptr<GLFence>&&);
    virtual ~CoordinatedPlatformLayerBufferExternalOES();

private:
    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) override;

    unsigned m_textureID { 0 };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_COORDINATED_PLATFORM_LAYER_BUFFER_TYPE(CoordinatedPlatformLayerBufferExternalOES, Type::ExternalOES)

#endif // USE(COORDINATED_GRAPHICS)
