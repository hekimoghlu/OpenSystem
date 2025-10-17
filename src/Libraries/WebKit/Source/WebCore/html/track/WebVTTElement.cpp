/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#include "WebVTTElement.h"

#if ENABLE(VIDEO)

#include "ElementInlines.h"
#include "HTMLSpanElement.h"
#include "RenderTreePosition.h"
#include "TextTrack.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebVTTElement);

static const QualifiedName& nodeTypeToTagName(WebVTTNodeType nodeType)
{
    static NeverDestroyed<QualifiedName> cTag(nullAtom(), "c"_s, nullAtom());
    static NeverDestroyed<QualifiedName> vTag(nullAtom(), "v"_s, nullAtom());
    static NeverDestroyed<QualifiedName> langTag(nullAtom(), "lang"_s, nullAtom());
    static NeverDestroyed<QualifiedName> bTag(nullAtom(), "b"_s, nullAtom());
    static NeverDestroyed<QualifiedName> uTag(nullAtom(), "u"_s, nullAtom());
    static NeverDestroyed<QualifiedName> iTag(nullAtom(), "i"_s, nullAtom());
    static NeverDestroyed<QualifiedName> rubyTag(nullAtom(), "ruby"_s, nullAtom());
    static NeverDestroyed<QualifiedName> rtTag(nullAtom(), "rt"_s, nullAtom());
    switch (nodeType) {
    case WebVTTNodeTypeClass:
        return cTag;
    case WebVTTNodeTypeItalic:
        return iTag;
    case WebVTTNodeTypeLanguage:
        return langTag;
    case WebVTTNodeTypeBold:
        return bTag;
    case WebVTTNodeTypeUnderline:
        return uTag;
    case WebVTTNodeTypeRuby:
        return rubyTag;
    case WebVTTNodeTypeRubyText:
        return rtTag;
    case WebVTTNodeTypeVoice:
        return vTag;
    case WebVTTNodeTypeNone:
    default:
        ASSERT_NOT_REACHED();
        return cTag; // Make the compiler happy.
    }
}

WebVTTElement::WebVTTElement(WebVTTNodeType nodeType, AtomString language, Document& document)
    : Element(nodeTypeToTagName(nodeType), document, { })
    , m_webVTTNodeType(nodeType)
    , m_language(language)
{
}

Ref<Element> WebVTTElement::create(WebVTTNodeType nodeType, AtomString language, Document& document)
{
    return adoptRef(*new WebVTTElement(nodeType, language, document));
}

Ref<Element> WebVTTElement::cloneElementWithoutAttributesAndChildren(TreeScope& treeScope)
{
    Ref document = treeScope.documentScope();
    return create(m_webVTTNodeType, m_language, document);
}

Ref<HTMLElement> WebVTTElement::createEquivalentHTMLElement(Document& document)
{
    RefPtr<HTMLElement> htmlElement;

    switch (m_webVTTNodeType) {
    case WebVTTNodeTypeClass:
    case WebVTTNodeTypeLanguage:
    case WebVTTNodeTypeVoice:
        htmlElement = HTMLSpanElement::create(document);
        htmlElement->setAttributeWithoutSynchronization(HTMLNames::titleAttr, attributeWithoutSynchronization(voiceAttributeName()));
        htmlElement->setAttributeWithoutSynchronization(HTMLNames::langAttr, attributeWithoutSynchronization(langAttributeName()));
        break;
    case WebVTTNodeTypeItalic:
        htmlElement = HTMLElement::create(HTMLNames::iTag, document);
        break;
    case WebVTTNodeTypeBold:
        htmlElement = HTMLElement::create(HTMLNames::bTag, document);
        break;
    case WebVTTNodeTypeUnderline:
        htmlElement = HTMLElement::create(HTMLNames::uTag, document);
        break;
    case WebVTTNodeTypeRuby:
        htmlElement = HTMLElement::create(HTMLNames::rubyTag, document);
        break;
    case WebVTTNodeTypeRubyText:
        htmlElement = HTMLElement::create(HTMLNames::rtTag, document);
        break;
    case WebVTTNodeTypeNone:
        ASSERT_NOT_REACHED();
        break;
    }

    ASSERT(htmlElement);
    if (htmlElement)
        htmlElement->setAttributeWithoutSynchronization(HTMLNames::classAttr, attributeWithoutSynchronization(HTMLNames::classAttr));
    return htmlElement.releaseNonNull();
}

} // namespace WebCore

#endif
