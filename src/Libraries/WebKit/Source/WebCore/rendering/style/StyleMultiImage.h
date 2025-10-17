/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

#include "StyleImage.h"
#include "StyleInvalidImage.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Document;

struct ImageWithScale {
    RefPtr<StyleImage> image { StyleInvalidImage::create() };
    float scaleFactor { 1 };
    String mimeType { String() };
};

inline bool operator==(const ImageWithScale& a, const ImageWithScale& b)
{
    return a.image == b.image && a.scaleFactor == b.scaleFactor;
}

class StyleMultiImage : public StyleImage {
    WTF_MAKE_TZONE_ALLOCATED(StyleMultiImage);
public:
    virtual ~StyleMultiImage();

protected:
    StyleMultiImage(Type);

    bool equals(const StyleMultiImage& other) const;

    virtual ImageWithScale selectBestFitImage(const Document&) = 0;
    CachedImage* cachedImage() const final;

private:
    WrappedImagePtr data() const final;

    bool canRender(const RenderElement*, float multiplier) const final;
    bool isPending() const final;
    void load(CachedResourceLoader&, const ResourceLoaderOptions&) final;
    bool isLoaded(const RenderElement*) const final;
    bool errorOccurred() const final;
    FloatSize imageSize(const RenderElement*, float multiplier) const final;
    bool imageHasRelativeWidth() const final;
    bool imageHasRelativeHeight() const final;
    void computeIntrinsicDimensions(const RenderElement*, Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio) final;
    bool usesImageContainerSize() const final;
    void setContainerContextForRenderer(const RenderElement&, const FloatSize&, float);
    void addClient(RenderElement&) final;
    void removeClient(RenderElement&) final;
    bool hasClient(RenderElement&) const final;
    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    float imageScaleFactor() const final;
    bool knownToBeOpaque(const RenderElement&) const final;
    const StyleImage* selectedImage() const final { return m_selectedImage.get(); }
    StyleImage* selectedImage() final { return m_selectedImage.get(); }

    RefPtr<StyleImage> m_selectedImage;
    bool m_isPending { true };
};

} // namespace WebCore
