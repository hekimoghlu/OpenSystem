/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef LegacyTileGridTile_h
#define LegacyTileGridTile_h

#if PLATFORM(IOS_FAMILY)

#include "IntRect.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>

@class LegacyTileLayer;

namespace WebCore {

class LegacyTileGrid;

// Refcount the tiles so they work nicely in vector and we know when to remove the tile layer from the parent.
class LegacyTileGridTile : public RefCounted<LegacyTileGridTile> {
public:
    static Ref<LegacyTileGridTile> create(LegacyTileGrid* grid, const IntRect& rect)
    {
        return adoptRef<LegacyTileGridTile>(*new LegacyTileGridTile(grid, rect));
    }
    ~LegacyTileGridTile();

    LegacyTileLayer* tileLayer() const { return m_tileLayer.get(); }
    void invalidateRect(const IntRect& rectInSurface);
    IntRect rect() const { return m_rect; }
    void setRect(const IntRect& tileRect);
    void showBorder(bool);

private:
    LegacyTileGridTile(LegacyTileGrid*, const IntRect&);

    LegacyTileGrid* m_tileGrid;
    RetainPtr<LegacyTileLayer> m_tileLayer;
    IntRect m_rect;
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
#endif // LegacyTileGridTile_h
