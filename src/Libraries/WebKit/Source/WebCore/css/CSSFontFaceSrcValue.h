/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#include <wtf/Vector.h>

namespace WebCore {

class CachedFont;
class FontLoadRequest;
class SVGFontFaceElement;
class ScriptExecutionContext;
class WeakPtrImplWithEventTargetData;

class CSSFontFaceSrcLocalValue final : public CSSValue {
public:
    static Ref<CSSFontFaceSrcLocalValue> create(AtomString fontFaceName);
    ~CSSFontFaceSrcLocalValue();

    bool isEmpty() const { return m_fontFaceName.isEmpty(); }
    const AtomString& fontFaceName() const { return m_fontFaceName; }

    SVGFontFaceElement* svgFontFaceElement() const;
    void setSVGFontFaceElement(SVGFontFaceElement&);

    String customCSSText() const;
    bool equals(const CSSFontFaceSrcLocalValue&) const;

private:
    explicit CSSFontFaceSrcLocalValue(AtomString&&);

    AtomString m_fontFaceName;
    WeakPtr<SVGFontFaceElement, WeakPtrImplWithEventTargetData> m_element;
};

enum class FontTechnology : uint8_t {
    ColorColrv0,
    ColorColrv1,
    ColorCbdt,
    ColorSbix,
    ColorSvg,
    FeaturesAat,
    FeaturesGraphite,
    FeaturesOpentype,
    Incremental,
    Palettes,
    Variations,
    // Reserved for invalid conversion result.
    Invalid
};

inline ASCIILiteral cssTextFromFontTech(FontTechnology tech)
{
    switch (tech) {
    case FontTechnology::ColorColrv0:
        return "color-colrv0"_s;
    case FontTechnology::ColorColrv1:
        return "color-colrv1"_s;
    case FontTechnology::ColorCbdt:
        return "color-cbdt"_s;
    case FontTechnology::ColorSbix:
        return "color-sbix"_s;
    case FontTechnology::ColorSvg:
        return "color-svg"_s;
    case FontTechnology::FeaturesAat:
        return "features-aat"_s;
    case FontTechnology::FeaturesGraphite:
        return "features-graphite"_s;
    case FontTechnology::FeaturesOpentype:
        return "features-opentype"_s;
    case FontTechnology::Incremental:
        return "incremental"_s;
    case FontTechnology::Palettes:
        return "palettes"_s;
    case FontTechnology::Variations:
        return "variations"_s;
    default:
        return ""_s;
    }
}

class CSSFontFaceSrcResourceValue final : public CSSValue {
public:

    static Ref<CSSFontFaceSrcResourceValue> create(ResolvedURL, String format, Vector<FontTechnology>&& technologies, LoadedFromOpaqueSource = LoadedFromOpaqueSource::No);

    bool isEmpty() const { return m_location.specifiedURLString.isEmpty(); }
    std::unique_ptr<FontLoadRequest> fontLoadRequest(ScriptExecutionContext&, bool isInitiatingElementInUserAgentShadowTree);

    String customCSSText() const;
    bool customTraverseSubresources(const Function<bool(const CachedResource&)>&) const;
    void customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>&);
    void customClearReplacementURLForSubresources();
    bool customMayDependOnBaseURL() const;
    bool equals(const CSSFontFaceSrcResourceValue&) const;

private:
    explicit CSSFontFaceSrcResourceValue(ResolvedURL&&, String&& format, Vector<FontTechnology>&& technologies, LoadedFromOpaqueSource);

    ResolvedURL m_location;
    String m_format;
    Vector<FontTechnology> m_technologies;
    LoadedFromOpaqueSource m_loadedFromOpaqueSource { LoadedFromOpaqueSource::No };
    CachedResourceHandle<CachedFont> m_cachedFont;
    String m_replacementURLString;
    bool m_shouldUseResolvedURLInCSSText { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFontFaceSrcLocalValue, isFontFaceSrcLocalValue())
SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFontFaceSrcResourceValue, isFontFaceSrcResourceValue())
