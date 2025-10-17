/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#if USE(TEXTURE_MAPPER)

#include "TransformationMatrix.h"

namespace WebCore {

class Color;
class TextureMapper;

class TextureMapperPlatformLayer {
public:
    class Client {
    public:
        virtual void platformLayerWillBeDestroyed() = 0;
        virtual void setPlatformLayerNeedsDisplay() = 0;
    };

    TextureMapperPlatformLayer() = default;
    virtual ~TextureMapperPlatformLayer() = default;

    virtual void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix& modelViewMatrix = TransformationMatrix(), float opacity = 1.0) = 0;

    WEBCORE_EXPORT virtual void drawBorder(TextureMapper&, const Color&, float borderWidth, const FloatRect&, const TransformationMatrix&);

    void setClient(TextureMapperPlatformLayer::Client* client) { m_client = client; }

    virtual bool isHolePunchBuffer() const { return false; }
    virtual void notifyVideoPosition(const FloatRect&, const TransformationMatrix&) { };
    virtual void paintTransparentRectangle(TextureMapper&, const FloatRect&, const TransformationMatrix&) { };

protected:
    TextureMapperPlatformLayer::Client* client() { return m_client; }

private:
    TextureMapperPlatformLayer::Client* m_client { nullptr };
};

} // namespace WebCore

#endif // USE(TEXTURE_MAPPER)
