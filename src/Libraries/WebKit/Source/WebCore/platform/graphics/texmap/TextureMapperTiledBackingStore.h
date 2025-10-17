/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#ifndef TextureMapperTiledBackingStore_h
#define TextureMapperTiledBackingStore_h

#include "FloatRect.h"
#include "Image.h"
#include "TextureMapperBackingStore.h"
#include "TextureMapperTile.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class TextureMapper;

class TextureMapperTiledBackingStore : public RefCounted<TextureMapperTiledBackingStore>, public TextureMapperBackingStore {
public:
    static Ref<TextureMapperTiledBackingStore> create() { return adoptRef(*new TextureMapperTiledBackingStore); }
    virtual ~TextureMapperTiledBackingStore() = default;

    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix&, float) override;
    void drawBorder(TextureMapper&, const Color&, float borderWidth, const FloatRect&, const TransformationMatrix&) override;
    void drawRepaintCounter(TextureMapper&, int repaintCount, const Color&, const FloatRect&, const TransformationMatrix&) override;

    void updateContentsScale(float);
    void updateContents(TextureMapper&, Image*, const FloatSize&, const IntRect&);
    void updateContents(TextureMapper&, GraphicsLayer*, const FloatSize&, const IntRect&);

    void setContentsToImage(Image* image) { m_image = image; }

private:
    TextureMapperTiledBackingStore() = default;

    void createOrDestroyTilesIfNeeded(const FloatSize& backingStoreSize, const IntSize& tileSize, bool hasAlpha);
    void updateContentsFromImageIfNeeded(TextureMapper&);
    TransformationMatrix adjustedTransformForRect(const FloatRect&);
    inline FloatRect rect() const
    {
        FloatRect rect(FloatPoint::zero(), m_size);
        rect.scale(m_contentsScale);
        return rect;
    }

    Vector<TextureMapperTile> m_tiles;
    FloatSize m_size;
    RefPtr<Image> m_image;
    float m_contentsScale { 1 };
    bool m_isScaleDirty { false };
};

} // namespace WebCore

#endif
