/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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

#include "CSSParserContext.h"
#include "CSSValue.h"
#include "CachedResourceHandle.h"
#include "ResourceLoaderOptions.h"
#include <wtf/Function.h>
#include <wtf/Ref.h>

namespace WebCore {

class CachedImage;
class CachedResourceLoader;
class DeprecatedCSSOMValue;
class CSSStyleDeclaration;
class RenderElement;
class StyleImage;

namespace Style {
class BuilderState;
}

class CSSImageValue final : public CSSValue {
public:
    static Ref<CSSImageValue> create();
    static Ref<CSSImageValue> create(ResolvedURL, LoadedFromOpaqueSource, AtomString = { });
    static Ref<CSSImageValue> create(URL, LoadedFromOpaqueSource, AtomString = { });
    ~CSSImageValue();

    bool isPending() const;
    CachedImage* loadImage(CachedResourceLoader&, const ResourceLoaderOptions&);
    CachedImage* cachedImage() const { return m_cachedImage ? m_cachedImage.value().get() : nullptr; }

    // Take care when using this, and read https://drafts.csswg.org/css-values/#relative-urls
    const URL& imageURL() const { return m_location.resolvedURL; }

    URL reresolvedURL(const Document&) const;

    String customCSSText() const;

    Ref<DeprecatedCSSOMValue> createDeprecatedCSSOMWrapper(CSSStyleDeclaration&) const;

    bool customTraverseSubresources(const Function<bool(const CachedResource&)>&) const;
    void customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>&);
    void customClearReplacementURLForSubresources();
    bool customMayDependOnBaseURL() const;

    bool equals(const CSSImageValue&) const;

    bool knownToBeOpaque(const RenderElement&) const;

    RefPtr<StyleImage> createStyleImage(const Style::BuilderState&) const;

    bool isLoadedFromOpaqueSource() const { return m_loadedFromOpaqueSource == LoadedFromOpaqueSource::Yes; }

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (m_unresolvedValue) {
            if (func(*m_unresolvedValue) == IterationStatus::Done)
                return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    }

private:
    CSSImageValue();
    CSSImageValue(ResolvedURL&&, LoadedFromOpaqueSource, AtomString&&);

    ResolvedURL m_location;
    std::optional<CachedResourceHandle<CachedImage>> m_cachedImage;
    AtomString m_initiatorType;
    LoadedFromOpaqueSource m_loadedFromOpaqueSource { LoadedFromOpaqueSource::No };
    RefPtr<CSSImageValue> m_unresolvedValue;
    bool m_isInvalid { false };
    String m_replacementURLString;
    bool m_shouldUseResolvedURLInCSSText { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSImageValue, isImageValue())
