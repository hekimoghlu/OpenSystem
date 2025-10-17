/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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

#if USE(GRAPHICS_LAYER_WC)

#include "IntPointHash.h"
#include "TextureMapperBackingStore.h"
#include "TextureMapperTile.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class TextureMapperSparseBackingStore final : public TextureMapperBackingStore {
    WTF_MAKE_TZONE_ALLOCATED(TextureMapperSparseBackingStore);
public:
    using TileIndex = WebCore::IntPoint;

    WEBCORE_EXPORT void setSize(const IntSize&);
    WEBCORE_EXPORT void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix&, float) override;
    WEBCORE_EXPORT void drawBorder(TextureMapper&, const Color&, float borderWidth, const FloatRect&, const TransformationMatrix&) override;
    WEBCORE_EXPORT void drawRepaintCounter(TextureMapper&, int repaintCount, const Color&, const FloatRect&, const TransformationMatrix&) override;
    WEBCORE_EXPORT void updateContents(const TileIndex&, Image&, const IntRect& dirtyRect);
    WEBCORE_EXPORT void removeTile(const TileIndex&);

private:
    TransformationMatrix adjustedTransformForRect(const FloatRect&);

    IntSize m_size;
    UncheckedKeyHashMap<TileIndex, std::unique_ptr<TextureMapperTile>> m_tiles;
};

} // namespace WebCore

#endif // USE(GRAPHICS_LAYER_WC)
