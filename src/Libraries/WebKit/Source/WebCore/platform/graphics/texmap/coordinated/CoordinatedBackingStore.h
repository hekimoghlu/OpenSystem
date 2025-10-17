/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#include "CoordinatedBackingStoreTile.h"
#include "TextureMapperBackingStore.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>

namespace WebCore {
class CoordinatedTileBuffer;
class TextureMapper;

class CoordinatedBackingStore final : public RefCounted<CoordinatedBackingStore>, public TextureMapperBackingStore {
public:
    static Ref<CoordinatedBackingStore> create()
    {
        return adoptRef(*new CoordinatedBackingStore);
    }
    ~CoordinatedBackingStore() = default;

    void resize(const FloatSize&, float scale);

    void createTile(uint32_t tileID);
    void removeTile(uint32_t tileID);
    void updateTile(uint32_t tileID, const IntRect&, const IntRect&, RefPtr<CoordinatedTileBuffer>&&, const IntPoint&);

    void processPendingUpdates(TextureMapper&);

    void paintToTextureMapper(TextureMapper&, const FloatRect&, const TransformationMatrix&, float) override;
    void drawBorder(TextureMapper&, const Color&, float borderWidth, const FloatRect&, const TransformationMatrix&) override;
    void drawRepaintCounter(TextureMapper&, int repaintCount, const Color&, const FloatRect&, const TransformationMatrix&) override;

private:
    CoordinatedBackingStore() = default;

    UncheckedKeyHashMap<uint32_t, CoordinatedBackingStoreTile> m_tiles;
    FloatSize m_size;
    float m_scale { 1. };
};

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
