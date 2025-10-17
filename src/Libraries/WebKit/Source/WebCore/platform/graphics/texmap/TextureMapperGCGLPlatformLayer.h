/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#if ENABLE(WEBGL) && USE(TEXTURE_MAPPER) && !USE(COORDINATED_GRAPHICS)

#include "GraphicsContextGLTextureMapperANGLE.h"
#include "PlatformLayer.h"
#include "TextureMapperPlatformLayer.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class TextureMapperPlatformLayerProxy;

class TextureMapperGCGLPlatformLayer : public PlatformLayer {
    WTF_MAKE_TZONE_ALLOCATED(TextureMapperGCGLPlatformLayer);
public:
    TextureMapperGCGLPlatformLayer(GraphicsContextGLTextureMapperANGLE&);
    virtual ~TextureMapperGCGLPlatformLayer();

    virtual void paintToTextureMapper(TextureMapper&, const FloatRect& target, const TransformationMatrix&, float opacity);

private:
    GraphicsContextGLTextureMapperANGLE& m_context;
};

} // namespace WebCore

#endif // ENABLE(WEBGL) && USE(TEXTURE_MAPPER) && !USE(COORDINATED_GRAPHICS)
