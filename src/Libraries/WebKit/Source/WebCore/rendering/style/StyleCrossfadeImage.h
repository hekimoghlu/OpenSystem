/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

#include "CachedImageClient.h"
#include "CachedResourceHandle.h"
#include "StyleGeneratedImage.h"

namespace WebCore {

struct BlendingContext;

class StyleCrossfadeImage final : public StyleGeneratedImage, private CachedImageClient {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(StyleCrossfadeImage);
public:
    static Ref<StyleCrossfadeImage> create(RefPtr<StyleImage> from, RefPtr<StyleImage> to, double percentage, bool isPrefixed)
    {
        return adoptRef(*new StyleCrossfadeImage(WTFMove(from), WTFMove(to), percentage, isPrefixed));
    }
    virtual ~StyleCrossfadeImage();

    RefPtr<StyleCrossfadeImage> blend(const StyleCrossfadeImage&, const BlendingContext&) const;

    bool operator==(const StyleImage&) const final;
    bool equals(const StyleCrossfadeImage&) const;
    bool equalInputImages(const StyleCrossfadeImage&) const;

    static constexpr bool isFixedSize = true;

private:
    explicit StyleCrossfadeImage(RefPtr<StyleImage>&&, RefPtr<StyleImage>&&, double, bool);

    Ref<CSSValue> computedStyleValue(const RenderStyle&) const final;
    bool isPending() const final;
    void load(CachedResourceLoader&, const ResourceLoaderOptions&) final;
    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    bool knownToBeOpaque(const RenderElement&) const final;
    FloatSize fixedSize(const RenderElement&) const final;
    void didAddClient(RenderElement&) final { }
    void didRemoveClient(RenderElement&) final { }

    // CachedImageClient.
    void imageChanged(CachedImage*, const IntRect*) final;

    RefPtr<StyleImage> m_from;
    RefPtr<StyleImage> m_to;
    double m_percentage;
    bool m_isPrefixed;

    // FIXME: Rather than caching and tracking the input image via CachedImages, we should
    // instead use a new, StyleImage specific notification, to allow correct tracking of
    // nested images (e.g. one of the input images for a StyleCrossfadeImage is a StyleFilterImage
    // where its input image is a StyleCachedImage).
    CachedResourceHandle<CachedImage> m_cachedFromImage;
    CachedResourceHandle<CachedImage> m_cachedToImage;
    bool m_inputImagesAreReady;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StyleCrossfadeImage, isCrossfadeImage)
