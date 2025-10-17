/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "SVGURIReference.h"

#include "Document.h"
#include "Element.h"
#include "SVGElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGUseElement.h"
#include "XLinkNames.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGURIReference);

SVGURIReference::SVGURIReference(SVGElement* contextElement)
    : m_href(SVGAnimatedString::create(contextElement, IsHrefProperty::Yes))
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::hrefAttr, &SVGURIReference::m_href>();
        PropertyRegistry::registerProperty<XLinkNames::hrefAttr, &SVGURIReference::m_href>();
    });
}

bool SVGURIReference::isKnownAttribute(const QualifiedName& attributeName)
{
    return PropertyRegistry::isKnownAttribute(attributeName);
}

SVGElement& SVGURIReference::contextElement() const
{
    return *m_href->contextElement();
}

void SVGURIReference::parseAttribute(const QualifiedName& name, const AtomString& value)
{
    if (name.matches(SVGNames::hrefAttr))
        m_href->setBaseValInternal(value.isNull() ? contextElement().getAttribute(XLinkNames::hrefAttr) : value);
    else if (name.matches(XLinkNames::hrefAttr) && !contextElement().hasAttribute(SVGNames::hrefAttr))
        m_href->setBaseValInternal(value);
}

AtomString SVGURIReference::fragmentIdentifierFromIRIString(const String& url, const Document& document)
{
    size_t start = url.find('#');
    if (start == notFound)
        return emptyAtom();

    if (!start)
        return StringView(url).substring(1).toAtomString();

    URL base = URL(document.baseURL(), url.left(start));
    String fragmentIdentifier = url.substring(start);
    URL urlWithFragment(base, fragmentIdentifier);
    if (equalIgnoringFragmentIdentifier(urlWithFragment, document.url()))
        return StringView(fragmentIdentifier).substring(1).toAtomString();

    // The url doesn't have any fragment identifier.
    return emptyAtom();
}

auto SVGURIReference::targetElementFromIRIString(const String& iri, const TreeScope& treeScope, RefPtr<Document> externalDocument) -> TargetElementResult
{
    // If there's no fragment identifier contained within the IRI string, we can't lookup an element.
    size_t startOfFragmentIdentifier = iri.find('#');
    if (startOfFragmentIdentifier == notFound)
        return { };

    // Exclude the '#' character when determining the fragmentIdentifier.
    auto id = StringView(iri).substring(startOfFragmentIdentifier + 1).toAtomString();
    if (id.isEmpty())
        return { };

    Ref document = treeScope.documentScope();
    auto url = document->completeURL(iri);
    if (externalDocument) {
        // Enforce that the referenced url matches the url of the document that we've loaded for it!
        ASSERT(equalIgnoringFragmentIdentifier(url, externalDocument->url()));
        return { externalDocument->getElementById(id), WTFMove(id) };
    }

    if (url.protocolIsData()) {
        // FIXME: We need to load the data url in a Document to be able to get the target element.
        if (!equalIgnoringFragmentIdentifier(url, document->url()))
            return { nullptr, WTFMove(id) };
    }

    // Exit early if the referenced url is external, and we have no externalDocument given.
    if (isExternalURIReference(iri, document))
        return { nullptr, WTFMove(id) };

    RefPtr shadowHost = treeScope.rootNode().shadowHost();
    if (is<SVGUseElement>(shadowHost))
        return { shadowHost->treeScope().getElementById(id), WTFMove(id) };

    return { treeScope.getElementById(id), WTFMove(id) };
}

bool SVGURIReference::haveLoadedRequiredResources() const
{
    if (href().isEmpty())
        return true;
    if (contextElement().protectedDocument()->completeURL(href()).protocolIsData())
        return true;
    if (!isExternalURIReference(href(), contextElement().protectedDocument()))
        return true;
    return errorOccurred() || haveFiredLoadEvent();
}

void SVGURIReference::dispatchLoadEvent()
{
    if (haveFiredLoadEvent())
        return;

    // Dispatch the load event
    setHaveFiredLoadEvent(true);
    ASSERT(contextElement().haveLoadedRequiredResources());

    contextElement().sendLoadEventIfPossible();
}

}
