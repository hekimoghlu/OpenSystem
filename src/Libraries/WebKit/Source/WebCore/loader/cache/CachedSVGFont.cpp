/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#include "CachedSVGFont.h"

#include "CookieJar.h"
#include "ElementChildIteratorInlines.h"
#include "FontCreationContext.h"
#include "FontDescription.h"
#include "FontPlatformData.h"
#include "ParserContentPolicy.h"
#include "SVGDocument.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFontElement.h"
#include "SVGFontFaceElement.h"
#include "SVGToOTFFontConversion.h"
#include "ScriptDisallowedScope.h"
#include "Settings.h"
#include "SharedBuffer.h"
#include "TextResourceDecoder.h"
#include "TypedElementDescendantIteratorInlines.h"

namespace WebCore {

CachedSVGFont::CachedSVGFont(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar, const Settings& settings)
    : CachedFont(WTFMove(request), sessionID, cookieJar, Type::SVGFontResource)
    , m_settings(settings)
{
}

CachedSVGFont::CachedSVGFont(CachedResourceRequest&& request, CachedSVGFont& resource)
    : CachedSVGFont(WTFMove(request), resource.sessionID(), resource.protectedCookieJar().get(), resource.m_settings.copyRef())
{
}

CachedSVGFont::~CachedSVGFont() = default;

RefPtr<Font> CachedSVGFont::createFont(const FontDescription& fontDescription, bool syntheticBold, bool syntheticItalic, const FontCreationContext& fontCreationContext)
{
    ASSERT(firstFontFace());
    return CachedFont::createFont(fontDescription, syntheticBold, syntheticItalic, fontCreationContext);
}

FontPlatformData CachedSVGFont::platformDataFromCustomData(const FontDescription& fontDescription, bool bold, bool italic, const FontCreationContext& fontCreationContext)
{
    if (m_externalSVGDocument)
        return FontPlatformData(fontDescription.computedSize(), bold, italic); // FIXME: Why are we creating a bogus font here?
    return CachedFont::platformDataFromCustomData(fontDescription, bold, italic, fontCreationContext);
}

bool CachedSVGFont::ensureCustomFontData()
{
    if (!m_externalSVGDocument && !errorOccurred() && !isLoading() && m_data) {
        bool sawError = false;
        {
            // We may get here during render tree updates when events are forbidden.
            // Frameless document can't run scripts or call back to the client so this is safe.
            Ref externalSVGDocument = SVGDocument::create(nullptr, m_settings.copyRef(), URL());
            Ref decoder = TextResourceDecoder::create("application/xml"_s);

            ScriptDisallowedScope::DisableAssertionsInScope disabledScope;

            externalSVGDocument->setMarkupUnsafe(decoder->decodeAndFlush(m_data->makeContiguous()->span()), { ParserContentPolicy::AllowDeclarativeShadowRoots });
            sawError = decoder->sawError();
            m_externalSVGDocument = WTFMove(externalSVGDocument);
        }

        if (sawError)
            m_externalSVGDocument = nullptr;
        if (m_externalSVGDocument)
            maybeInitializeExternalSVGFontElement();
        if (!m_externalSVGFontElement || !firstFontFace())
            return false;
        if (auto convertedFont = convertSVGToOTFFont(Ref { *m_externalSVGFontElement }))
            m_convertedFont = SharedBuffer::create(WTFMove(convertedFont.value()));
        else {
            m_externalSVGDocument = nullptr;
            m_externalSVGFontElement = nullptr;
            return false;
        }
    }

    return m_externalSVGDocument && CachedFont::ensureCustomFontData(m_convertedFont.copyRef().get());
}

SVGFontElement* CachedSVGFont::getSVGFontById(const AtomString& fontName) const
{
    ASSERT(m_externalSVGDocument);
    auto elements = descendantsOfType<SVGFontElement>(*m_externalSVGDocument);

    if (fontName.isEmpty())
        return elements.first();

    for (auto& element : elements) {
        if (element.getIdAttribute() == fontName)
            return &element;
    }
    return nullptr;
}

SVGFontElement* CachedSVGFont::maybeInitializeExternalSVGFontElement()
{
    if (!m_externalSVGFontElement)
        m_externalSVGFontElement = getSVGFontById(url().fragmentIdentifier().toAtomString());
    return m_externalSVGFontElement.get();
}

SVGFontFaceElement* CachedSVGFont::firstFontFace()
{
    if (!maybeInitializeExternalSVGFontElement())
        return nullptr;

    if (auto* firstFontFace = childrenOfType<SVGFontFaceElement>(*m_externalSVGFontElement).first())
        return firstFontFace;
    return nullptr;
}

}
