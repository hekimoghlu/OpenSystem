/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include "CSSFontFaceSrcValue.h"

#include "CSSMarkup.h"
#include "CachedFont.h"
#include "CachedFontLoadRequest.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedResourceRequestInitiatorTypes.h"
#include "FontCustomPlatformData.h"
#include "SVGFontFaceElement.h"
#include "ScriptExecutionContext.h"
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSFontFaceSrcLocalValue::CSSFontFaceSrcLocalValue(AtomString&& fontFaceName)
    : CSSValue(ClassType::FontFaceSrcLocal)
    , m_fontFaceName(WTFMove(fontFaceName))
{
}

Ref<CSSFontFaceSrcLocalValue> CSSFontFaceSrcLocalValue::create(AtomString fontFaceName)
{
    return adoptRef(*new CSSFontFaceSrcLocalValue { WTFMove(fontFaceName) });
}

CSSFontFaceSrcLocalValue::~CSSFontFaceSrcLocalValue() = default;

SVGFontFaceElement* CSSFontFaceSrcLocalValue::svgFontFaceElement() const
{
    return m_element.get();
}

void CSSFontFaceSrcLocalValue::setSVGFontFaceElement(SVGFontFaceElement& element)
{
    m_element = &element;
}

String CSSFontFaceSrcLocalValue::customCSSText() const
{
    return makeString("local("_s, serializeString(m_fontFaceName), ')');
}

bool CSSFontFaceSrcLocalValue::equals(const CSSFontFaceSrcLocalValue& other) const
{
    return m_fontFaceName == other.m_fontFaceName;
}

CSSFontFaceSrcResourceValue::CSSFontFaceSrcResourceValue(ResolvedURL&& location, String&& format, Vector<FontTechnology>&& technologies, LoadedFromOpaqueSource source)
    : CSSValue(ClassType::FontFaceSrcResource)
    , m_location(WTFMove(location))
    , m_format(WTFMove(format))
    , m_technologies(WTFMove(technologies))
    , m_loadedFromOpaqueSource(source)
{
}

Ref<CSSFontFaceSrcResourceValue> CSSFontFaceSrcResourceValue::create(ResolvedURL location, String format, Vector<FontTechnology>&& technologies, LoadedFromOpaqueSource source)
{
    return adoptRef(*new CSSFontFaceSrcResourceValue { WTFMove(location), WTFMove(format), WTFMove(technologies), source });
}

std::unique_ptr<FontLoadRequest> CSSFontFaceSrcResourceValue::fontLoadRequest(ScriptExecutionContext& context, bool isInitiatingElementInUserAgentShadowTree)
{
    if (m_cachedFont)
        return makeUnique<CachedFontLoadRequest>(*m_cachedFont, context);

    bool isFormatSVG;
    if (m_format.isEmpty()) {
        // In order to avoid conflicts with the old WinIE style of font-face, if there is no format specified,
        // we check to see if the URL ends with .eot. We will not try to load those.
        if (m_location.resolvedURL.lastPathComponent().endsWithIgnoringASCIICase(".eot"_s) && !m_location.resolvedURL.protocolIsData())
            return nullptr;
        isFormatSVG = false;
    } else {
        isFormatSVG = equalLettersIgnoringASCIICase(m_format, "svg"_s);
        if (!FontCustomPlatformData::supportsFormat(m_format))
            return nullptr;
    }

    if (!m_technologies.isEmpty()) {
        for (auto technology : m_technologies) {
            if (!FontCustomPlatformData::supportsTechnology(technology))
                return nullptr;
        }
    }

    auto request = context.fontLoadRequest(m_location.resolvedURL.string(), isFormatSVG, isInitiatingElementInUserAgentShadowTree, m_loadedFromOpaqueSource);
    if (auto* cachedRequest = dynamicDowncast<CachedFontLoadRequest>(request.get()))
        m_cachedFont = &cachedRequest->cachedFont();

    return request;
}

bool CSSFontFaceSrcResourceValue::customTraverseSubresources(const Function<bool(const CachedResource&)>& handler) const
{
    return m_cachedFont && handler(*m_cachedFont);
}

void CSSFontFaceSrcResourceValue::customSetReplacementURLForSubresources(const UncheckedKeyHashMap<String, String>& replacementURLStrings)
{
    auto replacementURLString = replacementURLStrings.get(m_location.resolvedURL.string());
    if (!replacementURLString.isNull())
        m_replacementURLString = replacementURLString;
    m_shouldUseResolvedURLInCSSText = true;
}

void CSSFontFaceSrcResourceValue::customClearReplacementURLForSubresources()
{
    m_replacementURLString = { };
    m_shouldUseResolvedURLInCSSText = false;
}

bool CSSFontFaceSrcResourceValue::customMayDependOnBaseURL() const
{
    return WebCore::mayDependOnBaseURL(m_location);
}

String CSSFontFaceSrcResourceValue::customCSSText() const
{
    StringBuilder builder;
    if (!m_replacementURLString.isEmpty())
        builder.append(serializeURL(m_replacementURLString));
    else {
        if (m_shouldUseResolvedURLInCSSText)
            builder.append(serializeURL(m_location.resolvedURL.string()));
        else
            builder.append(serializeURL(m_location.specifiedURLString));
    }
    if (!m_format.isEmpty())
        builder.append(" format("_s, serializeString(m_format), ')');
    if (!m_technologies.isEmpty()) {
        builder.append(" tech("_s);
        for (size_t i = 0; i < m_technologies.size(); ++i) {
            if (i)
                builder.append(", "_s);
            builder.append(cssTextFromFontTech(m_technologies[i]));
        }
        builder.append(')');
    }
    return builder.toString();
}

bool CSSFontFaceSrcResourceValue::equals(const CSSFontFaceSrcResourceValue& other) const
{
    return m_location == other.m_location
        && m_format == other.m_format
        && m_technologies == other.m_technologies
        && m_loadedFromOpaqueSource == other.m_loadedFromOpaqueSource;
}

}
