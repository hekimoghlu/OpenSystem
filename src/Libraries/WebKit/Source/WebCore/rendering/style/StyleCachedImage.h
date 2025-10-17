/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#include "CachedResourceHandle.h"
#include "StyleImage.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CSSValue;
class CSSImageValue;
class CachedImage;
class Document;
class LegacyRenderSVGResourceContainer;
class RenderElement;
class RenderSVGResourceContainer;
class TreeScope;

class StyleCachedImage final : public StyleImage {
    WTF_MAKE_TZONE_ALLOCATED(StyleCachedImage);
public:
    static Ref<StyleCachedImage> create(Ref<CSSImageValue>, float scaleFactor = 1);
    static Ref<StyleCachedImage> copyOverridingScaleFactor(StyleCachedImage&, float scaleFactor);
    virtual ~StyleCachedImage();

    bool operator==(const StyleImage&) const final;
    bool equals(const StyleCachedImage&) const;

    CachedImage* cachedImage() const final;

    WrappedImagePtr data() const final { return m_cachedImage.get(); }

    Ref<CSSValue> computedStyleValue(const RenderStyle&) const final;
    
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
    void setContainerContextForRenderer(const RenderElement&, const FloatSize&, float) final;
    void addClient(RenderElement&) final;
    void removeClient(RenderElement&) final;
    bool hasClient(RenderElement&) const final;
    bool hasImage() const final;
    RefPtr<Image> image(const RenderElement*, const FloatSize&, bool isForFirstLine) const final;
    float imageScaleFactor() const final;
    bool knownToBeOpaque(const RenderElement&) const final;
    bool usesDataProtocol() const final;

    URL reresolvedURL(const Document&) const;

    URL imageURL() const;

private:
    StyleCachedImage(Ref<CSSImageValue>&&, float);

    LegacyRenderSVGResourceContainer* uncheckedRenderSVGResource(TreeScope&, const AtomString& fragment) const;
    LegacyRenderSVGResourceContainer* uncheckedRenderSVGResource(const RenderElement*) const;
    LegacyRenderSVGResourceContainer* legacyRenderSVGResource(const RenderElement*) const;
    RenderSVGResourceContainer* renderSVGResource(const RenderElement*) const;
    bool isRenderSVGResource(const RenderElement*) const;

    Ref<CSSImageValue> m_cssValue;
    bool m_isPending { true };
    mutable float m_scaleFactor { 1 };
    mutable CachedResourceHandle<CachedImage> m_cachedImage;
    mutable std::optional<bool> m_isRenderSVGResource;
    FloatSize m_containerSize;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_STYLE_IMAGE(StyleCachedImage, isCachedImage)
