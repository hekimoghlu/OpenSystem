/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#include "HTMLDocument.h"

#include "CSSPropertyNames.h"
#include "CommonVM.h"
#include "CookieJar.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "DocumentType.h"
#include "ElementChildIteratorInlines.h"
#include "FocusController.h"
#include "FrameLoader.h"
#include "FrameTree.h"
#include "HTMLBodyElement.h"
#include "HTMLCollection.h"
#include "HTMLDocumentParser.h"
#include "HTMLElementFactory.h"
#include "HTMLFrameOwnerElement.h"
#include "HTMLFrameSetElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLIFrameElement.h"
#include "HTMLNames.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Quirks.h"
#include "ScriptController.h"
#include "StyleResolver.h"
#include <wtf/RobinHoodHashSet.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLDocument);

using namespace HTMLNames;

Ref<HTMLDocument> HTMLDocument::createSynthesizedDocument(LocalFrame& frame, const URL& url)
{
    auto document = adoptRef(*new HTMLDocument(&frame, frame.settings(), url, { }, { DocumentClass::HTML }, { ConstructionFlag::Synthesized }));
    document->addToContextsMap();
    return document;
}

HTMLDocument::HTMLDocument(LocalFrame* frame, const Settings& settings, const URL& url, std::optional<ScriptExecutionContextIdentifier> documentIdentifier, DocumentClasses documentClasses, OptionSet<ConstructionFlag> constructionFlags)
    : Document(frame, settings, url, documentClasses | DocumentClasses(DocumentClass::HTML), constructionFlags, documentIdentifier)
{
    clearXMLVersion();
}

HTMLDocument::~HTMLDocument() = default;

Ref<DocumentParser> HTMLDocument::createParser()
{
    return HTMLDocumentParser::create(*this, parserContentPolicy());
}

// https://html.spec.whatwg.org/multipage/dom.html#dom-document-nameditem
std::optional<std::variant<RefPtr<WindowProxy>, RefPtr<Element>, RefPtr<HTMLCollection>>> HTMLDocument::namedItem(const AtomString& name)
{
    if (name.isNull() || !hasDocumentNamedItem(name))
        return std::nullopt;

    if (UNLIKELY(documentNamedItemContainsMultipleElements(name))) {
        auto collection = documentNamedItems(name);
        ASSERT(collection->length() > 1);
        return std::variant<RefPtr<WindowProxy>, RefPtr<Element>, RefPtr<HTMLCollection>> { RefPtr<HTMLCollection> { WTFMove(collection) } };
    }

    Ref element = *documentNamedItem(name);
    if (auto* iframe = dynamicDowncast<HTMLIFrameElement>(element.get()); UNLIKELY(iframe)) {
        if (RefPtr domWindow = iframe->contentWindow())
            return std::variant<RefPtr<WindowProxy>, RefPtr<Element>, RefPtr<HTMLCollection>> { WTFMove(domWindow) };
    }

    return std::variant<RefPtr<WindowProxy>, RefPtr<Element>, RefPtr<HTMLCollection>> { RefPtr<Element> { WTFMove(element) } };
}

bool HTMLDocument::isSupportedPropertyName(const AtomString& name) const
{
    return !name.isNull() && hasDocumentNamedItem(name);
}

Vector<AtomString> HTMLDocument::supportedPropertyNames() const
{
    if (Quirks::shouldOmitHTMLDocumentSupportedPropertyNames())
        return { };

    auto properties = m_documentNamedItem.keys();
    // The specification says these should be sorted in document order but this would be expensive
    // and other browser engines do not comply with this part of the specification. For now, just
    // do an alphabetical sort to get consistent results.
    std::sort(properties.begin(), properties.end(), WTF::codePointCompareLessThan);
    return properties;
}

void HTMLDocument::addDocumentNamedItem(const AtomString& name, Element& item)
{
    m_documentNamedItem.add(name, item, *this);
    addImpureProperty(name);
}

void HTMLDocument::removeDocumentNamedItem(const AtomString& name, Element& item)
{
    m_documentNamedItem.remove(name, item);
}

void HTMLDocument::addWindowNamedItem(const AtomString& name, Element& item)
{
    m_windowNamedItem.add(name, item, *this);
}

void HTMLDocument::removeWindowNamedItem(const AtomString& name, Element& item)
{
    m_windowNamedItem.remove(name, item);
}

bool HTMLDocument::isCaseSensitiveAttribute(const QualifiedName& attributeName)
{
    static NeverDestroyed set = [] {
        // https://html.spec.whatwg.org/multipage/semantics-other.html#case-sensitivity-of-selectors
        static constexpr std::array names {
            &accept_charsetAttr,
            &acceptAttr,
            &alignAttr,
            &alinkAttr,
            &axisAttr,
            &bgcolorAttr,
            &charsetAttr,
            &checkedAttr,
            &clearAttr,
            &codetypeAttr,
            &colorAttr,
            &compactAttr,
            &declareAttr,
            &deferAttr,
            &dirAttr,
            &directionAttr,
            &disabledAttr,
            &enctypeAttr,
            &faceAttr,
            &frameAttr,
            &hreflangAttr,
            &http_equivAttr,
            &langAttr,
            &languageAttr,
            &linkAttr,
            &mediaAttr,
            &methodAttr,
            &multipleAttr,
            &nohrefAttr,
            &noresizeAttr,
            &noshadeAttr,
            &nowrapAttr,
            &readonlyAttr,
            &relAttr,
            &revAttr,
            &rulesAttr,
            &scopeAttr,
            &scrollingAttr,
            &selectedAttr,
            &shapeAttr,
            &targetAttr,
            &textAttr,
            &typeAttr,
            &valignAttr,
            &valuetypeAttr,
            &vlinkAttr,
        };
        MemoryCompactLookupOnlyRobinHoodHashSet<AtomString> set;
        set.reserveInitialCapacity(std::size(names));
        for (auto* name : names)
            set.add(name->get().localName());
        return set;
    }();
    auto isPossibleHTMLAttr = !attributeName.hasPrefix() && attributeName.namespaceURI().isNull();
    return !isPossibleHTMLAttr || !set.get().contains(attributeName.localName());
}

bool HTMLDocument::isFrameSet() const
{
    if (!documentElement())
        return false;
    return !!childrenOfType<HTMLFrameSetElement>(*documentElement()).first();
}

Ref<Document> HTMLDocument::cloneDocumentWithoutChildren() const
{
    return create(nullptr, settings(), url());
}

}
