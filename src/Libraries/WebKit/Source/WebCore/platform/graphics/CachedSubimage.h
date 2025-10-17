/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include "ImageTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GraphicsContext;
class ImageBuffer;
struct ImagePaintingOptions;

class CachedSubimage {
    WTF_MAKE_TZONE_ALLOCATED(CachedSubimage);
public:
    static std::unique_ptr<CachedSubimage> create(GraphicsContext&, const FloatSize& imageSize, const FloatRect& destinationRect, const FloatRect& sourceRect);
    static std::unique_ptr<CachedSubimage> createPixelated(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect);

    CachedSubimage(Ref<ImageBuffer>&&, const FloatSize& scaleFactor, const FloatRect& destinationRect, const FloatRect& sourceRect);

    ImageBuffer& imageBuffer() const { return m_imageBuffer; }
    FloatSize scaleFactor() const { return m_scaleFactor; }
    FloatRect destinationRect() const { return m_destinationRect; }
    FloatRect sourceRect() const { return m_sourceRect; }

    bool canBeUsed(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect) const;
    void draw(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect);

    // The tile size is usually 512 x 512, account for the 2x display resolution
    // and make room for one more tile in every direction.
    static constexpr float maxSide = 512 * 2 * 3;
    static constexpr float maxArea = maxSide * maxSide;

private:
    Ref<ImageBuffer> m_imageBuffer;
    FloatSize m_scaleFactor;
    FloatRect m_destinationRect;
    FloatRect m_sourceRect;
};

} // namespace WebCore
