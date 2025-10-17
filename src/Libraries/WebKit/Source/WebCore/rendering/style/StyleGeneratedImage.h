/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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

#include "FloatSize.h"
#include "FloatSizeHash.h"
#include "StyleImage.h"
#include <wtf/HashMap.h>
#include <wtf/WeakHashCountedSet.h>

namespace WebCore {

class CSSValue;
class CachedImage;
class CachedResourceLoader;
class GeneratedImage;
class Image;
class RenderElement;

struct ResourceLoaderOptions;

class StyleGeneratedImage : public StyleImage {
public:
    const SingleThreadWeakHashCountedSet<RenderElement>& clients() const { return m_clients; }

protected:
    explicit StyleGeneratedImage(StyleImage::Type, bool fixedSize);
    virtual ~StyleGeneratedImage();

    WrappedImagePtr data() const final { return this; }

    FloatSize imageSize(const RenderElement*, float multiplier) const final;
    void computeIntrinsicDimensions(const RenderElement*, Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio) final;
    bool imageHasRelativeWidth() const final { return !m_fixedSize; }
    bool imageHasRelativeHeight() const final { return !m_fixedSize; }
    bool usesImageContainerSize() const final { return !m_fixedSize; }
    void setContainerContextForRenderer(const RenderElement&, const FloatSize& containerSize, float) final { m_containerSize = containerSize; }
    bool imageHasNaturalDimensions() const final { return !usesImageContainerSize(); }
    
    void addClient(RenderElement&) final;
    void removeClient(RenderElement&) final;
    bool hasClient(RenderElement&) const final;

    // Allow subclasses to react to clients being added/removed.
    virtual void didAddClient(RenderElement&) = 0;
    virtual void didRemoveClient(RenderElement&) = 0;

    // All generated images must be able to compute their fixed size.
    virtual FloatSize fixedSize(const RenderElement&) const = 0;

    class CachedGeneratedImage;
    GeneratedImage* cachedImageForSize(FloatSize);
    void saveCachedImageForSize(FloatSize, GeneratedImage&);
    void evictCachedGeneratedImage(FloatSize);

    FloatSize m_containerSize;
    bool m_fixedSize;
    SingleThreadWeakHashCountedSet<RenderElement> m_clients;
    UncheckedKeyHashMap<FloatSize, std::unique_ptr<CachedGeneratedImage>> m_images;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StyleGeneratedImage, isGeneratedImage)
