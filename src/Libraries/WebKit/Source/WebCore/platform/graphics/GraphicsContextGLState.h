/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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

#if ENABLE(WEBGL)

#include "GraphicsContextGL.h"
#include <wtf/HashCountedSet.h>
#include <wtf/HashMap.h>
#include <wtf/HashTraits.h>

namespace WebCore {

struct GraphicsContextGLState {
    GCGLuint boundReadFBO { 0 };
    GCGLuint boundDrawFBO { 0 };
    GCGLenum activeTextureUnit { GraphicsContextGL::TEXTURE0 };

    using BoundTextureMap = UncheckedKeyHashMap<GCGLenum,
        std::pair<GCGLuint, GCGLenum>,
        IntHash<GCGLenum>,
        WTF::UnsignedWithZeroKeyHashTraits<GCGLuint>,
        PairHashTraits<WTF::UnsignedWithZeroKeyHashTraits<GCGLuint>, WTF::UnsignedWithZeroKeyHashTraits<GCGLuint>>
    >;
    BoundTextureMap boundTextureMap;
    GCGLuint currentBoundTexture() const { return boundTexture(activeTextureUnit); }
    GCGLuint boundTexture(GCGLenum textureUnit) const
    {
        auto iterator = boundTextureMap.find(textureUnit);
        if (iterator != boundTextureMap.end())
            return iterator->value.first;
        return 0;
    }

    GCGLuint currentBoundTarget() const { return boundTarget(activeTextureUnit); }
    GCGLenum boundTarget(GCGLenum textureUnit) const
    {
        auto iterator = boundTextureMap.find(textureUnit);
        if (iterator != boundTextureMap.end())
            return iterator->value.second;
        return 0;
    }

    void setBoundTexture(GCGLenum textureUnit, GCGLuint texture, GCGLenum target)
    {
        boundTextureMap.set(textureUnit, std::make_pair(texture, target));
    }
};

}

#endif
