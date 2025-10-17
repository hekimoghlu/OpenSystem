/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#include "DocumentFragment.h"

#include "CSSTokenizerInputStream.h"
#include "Document.h"
#include "ElementIterator.h"
#include "HTMLDocumentParser.h"
#include "HTMLDocumentParserFastPath.h"
#include "Page.h"
#include "TypedElementDescendantIteratorInlines.h"
#include "XMLDocumentParser.h"
#include "markup.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DocumentFragment);

DocumentFragment::DocumentFragment(Document& document, OptionSet<TypeFlag> typeFlags)
    : ContainerNode(document, DOCUMENT_FRAGMENT_NODE, typeFlags)
{
}

Ref<DocumentFragment> DocumentFragment::create(Document& document)
{
    return adoptRef(*new DocumentFragment(document));
}

Ref<DocumentFragment> DocumentFragment::createForInnerOuterHTML(Document& document)
{
    auto node = adoptRef(*new DocumentFragment(document));
    node->setStateFlag(StateFlag::IsSpecialInternalNode);
    ASSERT(node->isDocumentFragmentForInnerOuterHTML());
    return node;
}

String DocumentFragment::nodeName() const
{
    return "#document-fragment"_s;
}

bool DocumentFragment::childTypeAllowed(NodeType type) const
{
    switch (type) {
        case ELEMENT_NODE:
        case PROCESSING_INSTRUCTION_NODE:
        case COMMENT_NODE:
        case TEXT_NODE:
        case CDATA_SECTION_NODE:
            return true;
        default:
            return false;
    }
}

Ref<Node> DocumentFragment::cloneNodeInternal(TreeScope& treeScope, CloningOperation type)
{
    Ref clone = create(treeScope.documentScope());
    switch (type) {
    case CloningOperation::OnlySelf:
    case CloningOperation::SelfWithTemplateContent:
        break;
    case CloningOperation::Everything:
        cloneChildNodes(treeScope, clone);
        break;
    }
    return clone;
}

void DocumentFragment::parseHTML(const String& source, Element& contextElement, OptionSet<ParserContentPolicy> parserContentPolicy, CustomElementRegistry* registry)
{
    Ref document = this->document();
    if (tryFastParsingHTMLFragment(source, document, *this, contextElement, parserContentPolicy)) {
#if ASSERT_ENABLED
        // As a sanity check for the fast-path, create another fragment using the full parser and compare the results.
        auto referenceFragment = DocumentFragment::create(document);
        HTMLDocumentParser::parseDocumentFragment(source, referenceFragment, contextElement, parserContentPolicy);
        ASSERT(serializeFragment(*this, SerializedNodes::SubtreesOfChildren) == serializeFragment(referenceFragment, SerializedNodes::SubtreesOfChildren));
#endif
        return;
    }
    if (hasChildNodes())
        removeChildren();

    HTMLDocumentParser::parseDocumentFragment(source, *this, contextElement, parserContentPolicy, registry);
}

bool DocumentFragment::parseXML(const String& source, Element* contextElement, OptionSet<ParserContentPolicy> parserContentPolicy)
{
    return XMLDocumentParser::parseDocumentFragment(source, *this, contextElement, parserContentPolicy);
}

Element* DocumentFragment::getElementById(const AtomString& id) const
{
    if (id.isEmpty())
        return nullptr;

    // Fast path for ShadowRoot, where we are both a DocumentFragment and a TreeScope.
    if (isTreeScope())
        return treeScope().getElementById(id).get();

    // Otherwise, fall back to iterating all of the element descendants.
    for (auto& element : descendantsOfType<Element>(*this)) {
        if (element.getIdAttribute() == id)
            return const_cast<Element*>(&element);
    }

    return nullptr;
}

}
