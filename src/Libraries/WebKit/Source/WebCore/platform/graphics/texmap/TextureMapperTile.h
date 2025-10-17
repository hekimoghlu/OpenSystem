/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

#include "FloatRect.h"
#include "Image.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class BitmapTexture;
class GraphicsLayer;
class TextureMapper;

class TextureMapperTile {
    WTF_MAKE_TZONE_ALLOCATED(TextureMapperTile);
public:
    RefPtr<BitmapTexture> texture() const;
    inline FloatRect rect() const { return m_rect; }
    void setTexture(BitmapTexture*);
    inline void setRect(const FloatRect& rect) { m_rect = rect; }

    void updateContents(Image*, const IntRect&);
    void updateContents(GraphicsLayer*, const IntRect&, float scale = 1);
    WEBCORE_EXPORT virtual void paint(TextureMapper&, const TransformationMatrix&, float, bool allEdgesExposed);
    virtual ~TextureMapperTile();

    explicit TextureMapperTile(const FloatRect&);
protected:
    RefPtr<BitmapTexture> m_texture;
private:
    FloatRect m_rect;
};

} // namespace WebCore
