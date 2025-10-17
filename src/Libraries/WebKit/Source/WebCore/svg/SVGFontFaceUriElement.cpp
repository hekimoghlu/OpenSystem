/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include "SVGFontFaceUriElement.h"

#include "CSSFontFaceSrcValue.h"
#include "CachedFont.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "Document.h"
#include "SVGElementInlines.h"
#include "SVGElementTypeHelpers.h"
#include "SVGFontFaceElement.h"
#include "SVGNames.h"
#include "XLinkNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFontFaceUriElement);
    
using namespace SVGNames;
    
inline SVGFontFaceUriElement::SVGFontFaceUriElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
{
    ASSERT(hasTagName(font_face_uriTag));
}

Ref<SVGFontFaceUriElement> SVGFontFaceUriElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFontFaceUriElement(tagName, document));
}

SVGFontFaceUriElement::~SVGFontFaceUriElement()
{
    if (CachedResourceHandle cachedFont = m_cachedFont)
        cachedFont->removeClient(*this);
}

Ref<CSSFontFaceSrcResourceValue> SVGFontFaceUriElement::createSrcValue() const
{
    ResolvedURL location;
    location.specifiedURLString = getAttribute(SVGNames::hrefAttr, XLinkNames::hrefAttr);
    if (!location.specifiedURLString.isNull())
        location.resolvedURL = document().completeURL(location.specifiedURLString);
    auto& format = attributeWithoutSynchronization(formatAttr);
    return CSSFontFaceSrcResourceValue::create(WTFMove(location), format.isEmpty() ? "svg"_s : format.string(), { FontTechnology::ColorSvg }, LoadedFromOpaqueSource::No);
}

void SVGFontFaceUriElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::hrefAttr || name == XLinkNames::hrefAttr)
        loadFont();

    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFontFaceUriElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);

    if (!parentNode() || !parentNode()->hasTagName(font_face_srcTag))
        return;
    
    if (RefPtr grandParent = dynamicDowncast<SVGFontFaceElement>(parentNode()->parentNode()))
        grandParent->rebuildFontFace();
}

Node::InsertedIntoAncestorResult SVGFontFaceUriElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    loadFont();
    return SVGElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
}

static bool isSVGFontTarget(const SVGFontFaceUriElement& element)
{
    auto& format = element.attributeWithoutSynchronization(formatAttr);
    return format.isEmpty() || equalLettersIgnoringASCIICase(format, "svg"_s);
}

void SVGFontFaceUriElement::loadFont()
{
    if (CachedResourceHandle cachedFont = m_cachedFont)
        cachedFont->removeClient(*this);

    const AtomString& href = getAttribute(SVGNames::hrefAttr, XLinkNames::hrefAttr);
    if (!href.isNull()) {
        ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
        options.contentSecurityPolicyImposition = isInUserAgentShadowTree() ? ContentSecurityPolicyImposition::SkipPolicyCheck : ContentSecurityPolicyImposition::DoPolicyCheck;

        Ref cachedResourceLoader = document().cachedResourceLoader();
        CachedResourceRequest request(ResourceRequest(document().completeURL(href)), options);
        request.setInitiator(*this);
        m_cachedFont = cachedResourceLoader->requestFont(WTFMove(request), isSVGFontTarget(*this)).value_or(nullptr);
        if (CachedResourceHandle cachedFont = m_cachedFont) {
            cachedFont->addClient(*this);
            cachedFont->beginLoadIfNeeded(cachedResourceLoader);
        }
    } else
        m_cachedFont = nullptr;
}

}
