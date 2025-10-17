/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include "config.h"
#include "StyleGeneratedImage.h"

#include "GeneratedImage.h"
#include "RenderElement.h"
#include "StyleResolver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static const Seconds timeToKeepCachedGeneratedImages { 3_s };

// MARK: - CachedGeneratedImage

class StyleGeneratedImage::CachedGeneratedImage {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CachedGeneratedImage);
public:
    CachedGeneratedImage(StyleGeneratedImage&, FloatSize, GeneratedImage&);
    GeneratedImage& image() const { return m_image; }
    void puntEvictionTimer() { m_evictionTimer.restart(); }

private:
    void evictionTimerFired();

    StyleGeneratedImage& m_owner;
    const FloatSize m_size;
    const Ref<GeneratedImage> m_image;
    DeferrableOneShotTimer m_evictionTimer;
};

inline StyleGeneratedImage::CachedGeneratedImage::CachedGeneratedImage(StyleGeneratedImage& owner, FloatSize size, GeneratedImage& image)
    : m_owner(owner)
    , m_size(size)
    , m_image(image)
    , m_evictionTimer(*this, &StyleGeneratedImage::CachedGeneratedImage::evictionTimerFired, timeToKeepCachedGeneratedImages)
{
    m_evictionTimer.restart();
}

void StyleGeneratedImage::CachedGeneratedImage::evictionTimerFired()
{
    // NOTE: This is essentially a "delete this", the object is no longer valid after this line.
    m_owner.evictCachedGeneratedImage(m_size);
}

// MARK: - StyleGeneratedImage.

StyleGeneratedImage::StyleGeneratedImage(StyleImage::Type type, bool fixedSize)
    : StyleImage { type }
    , m_fixedSize { fixedSize }
{
}

StyleGeneratedImage::~StyleGeneratedImage() = default;

GeneratedImage* StyleGeneratedImage::cachedImageForSize(FloatSize size)
{
    if (size.isEmpty())
        return nullptr;

    auto* cachedGeneratedImage = m_images.get(size);
    if (!cachedGeneratedImage)
        return nullptr;

    cachedGeneratedImage->puntEvictionTimer();
    return &cachedGeneratedImage->image();
}

void StyleGeneratedImage::saveCachedImageForSize(FloatSize size, GeneratedImage& image)
{
    ASSERT(!m_images.contains(size));
    m_images.add(size, makeUnique<CachedGeneratedImage>(*this, size, image));
}

void StyleGeneratedImage::evictCachedGeneratedImage(FloatSize size)
{
    ASSERT(m_images.contains(size));
    m_images.remove(size);
}

FloatSize StyleGeneratedImage::imageSize(const RenderElement* renderer, float multiplier) const
{
    if (!m_fixedSize)
        return m_containerSize;

    if (!renderer)
        return { };

    FloatSize fixedSize = this->fixedSize(*renderer);
    if (multiplier == 1.0f)
        return fixedSize;

    float width = fixedSize.width() * multiplier;
    float height = fixedSize.height() * multiplier;

    // Don't let images that have a width/height >= 1 shrink below 1 device pixel when zoomed.
    float deviceScaleFactor = renderer->document().deviceScaleFactor();
    if (fixedSize.width() > 0)
        width = std::max<float>(1 / deviceScaleFactor, width);
    if (fixedSize.height() > 0)
        height = std::max<float>(1 / deviceScaleFactor, height);

    return { width, height };
}

void StyleGeneratedImage::computeIntrinsicDimensions(const RenderElement* renderer, Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio)
{
    // At a zoom level of 1 the image is guaranteed to have a device pixel size.
    FloatSize size = floorSizeToDevicePixels(LayoutSize(this->imageSize(renderer, 1)), renderer ? renderer->document().deviceScaleFactor() : 1);
    intrinsicWidth = Length(size.width(), LengthType::Fixed);
    intrinsicHeight = Length(size.height(), LengthType::Fixed);
    intrinsicRatio = size;
}

// MARK: Client support.

void StyleGeneratedImage::addClient(RenderElement& renderer)
{
    if (m_clients.isEmptyIgnoringNullReferences())
        ref();

    m_clients.add(renderer);

    this->didAddClient(renderer);
}

void StyleGeneratedImage::removeClient(RenderElement& renderer)
{
    ASSERT(m_clients.contains(renderer));
    if (!m_clients.remove(renderer))
        return;

    this->didRemoveClient(renderer);

    if (m_clients.isEmptyIgnoringNullReferences())
        deref();
}

bool StyleGeneratedImage::hasClient(RenderElement& renderer) const
{
    return m_clients.contains(renderer);
}

} // namespace WebCore
