/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "FilterOperations.h"
#include "StyleGeneratedImage.h"

namespace WebCore {

class StyleFilterImage final : public StyleGeneratedImage, private CachedImageClient {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(StyleFilterImage);
public:
    static Ref<StyleFilterImage> create(RefPtr<StyleImage> image, FilterOperations filterOperations)
    {
        return adoptRef(*new StyleFilterImage(WTFMove(image), WTFMove(filterOperations)));
    }
    virtual ~StyleFilterImage();

    bool operator==(const StyleImage& other) const final;
    bool equals(const StyleFilterImage&) const;
    bool equalInputImages(const StyleFilterImage&) const;

    RefPtr<StyleImage> inputImage() const { return m_image; }
    const FilterOperations& filterOperations() const { return m_filterOperations; }

    static constexpr bool isFixedSize = true;

private:
    explicit StyleFilterImage(RefPtr<StyleImage>&&, FilterOperations&&);

    Ref<CSSValue> computedStyleValue(const RenderStyle&) const final;
    bool isPending() const final;
    void load(CachedResourceLoader&, const ResourceLoaderOptions&) final;
    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    bool knownToBeOpaque(const RenderElement&) const final;
    FloatSize fixedSize(const RenderElement&) const final;
    void didAddClient(RenderElement&) final { }
    void didRemoveClient(RenderElement&) final { }

    // CachedImageClient.
    void imageChanged(CachedImage*, const IntRect* = nullptr) final;

    RefPtr<StyleImage> m_image;
    FilterOperations m_filterOperations;

    // FIXME: Rather than caching and tracking the input image via CachedImages, we should
    // instead use a new, StyleImage specific notification, to allow correct tracking of
    // nested images (e.g. the input image for a StyleFilterImage is a StyleCrossfadeImage
    // where one of the inputs to the StyleCrossfadeImage is a StyleCachedImage).
    CachedResourceHandle<CachedImage> m_cachedImage;
    bool m_inputImageIsReady;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StyleFilterImage, isFilterImage)
