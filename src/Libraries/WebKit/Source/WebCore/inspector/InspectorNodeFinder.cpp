/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#include "InspectorNodeFinder.h"

#include "Attr.h"
#include "Document.h"
#include "ElementInlines.h"
#include "HTMLFrameOwnerElement.h"
#include "LocalFrame.h"
#include "NodeList.h"
#include "NodeTraversal.h"
#include "XPathNSResolver.h"
#include "XPathResult.h"

namespace WebCore {

static String stripCharacters(const String& string, const char startCharacter, const char endCharacter, bool& startCharFound, bool& endCharFound)
{
    startCharFound = string.startsWith(startCharacter);
    endCharFound = string.endsWith(endCharacter);

    unsigned start = startCharFound ? 1 : 0;
    unsigned end = string.length() - (endCharFound ? 1 : 0);
    return string.substring(start, end - start);
}

InspectorNodeFinder::InspectorNodeFinder(const String& query, bool caseSensitive)
    : m_query(query)
    , m_caseSensitive(caseSensitive)
{
    m_tagNameQuery = stripCharacters(query, '<', '>', m_startTagFound, m_endTagFound);

    bool startQuoteFound, endQuoteFound;
    m_attributeQuery = stripCharacters(query, '"', '"', startQuoteFound, endQuoteFound);
    m_exactAttributeMatch = startQuoteFound && endQuoteFound;
}

void InspectorNodeFinder::performSearch(Node* parentNode)
{
    if (!parentNode)
        return;

    searchUsingXPath(*parentNode);
    searchUsingCSSSelectors(*parentNode);

    // Keep the DOM tree traversal last. This way iframe content will come after their parents.
    searchUsingDOMTreeTraversal(*parentNode);
}

void InspectorNodeFinder::searchUsingDOMTreeTraversal(Node& parentNode)
{
    // Manual plain text search.
    for (auto* node = &parentNode; node; node = NodeTraversal::next(*node, &parentNode)) {
        switch (node->nodeType()) {
        case Node::TEXT_NODE:
        case Node::COMMENT_NODE:
        case Node::CDATA_SECTION_NODE:
            if (checkContains(node->nodeValue(), m_query))
                m_results.add(node);
            break;
        case Node::ELEMENT_NODE:
            if (matchesElement(downcast<Element>(*node)))
                m_results.add(node);
            if (auto* frameOwner = dynamicDowncast<HTMLFrameOwnerElement>(*node))
                performSearch(frameOwner->protectedContentDocument().get());
            break;
        default:
            break;
        }
    }
}

bool InspectorNodeFinder::checkEquals(const String& a, const String& b)
{
    if (m_caseSensitive)
        return a == b;
    return equalIgnoringASCIICase(a, b);
}

bool InspectorNodeFinder::checkContains(const String& a, const String& b)
{
    if (m_caseSensitive)
        return a.contains(b);
    return a.containsIgnoringASCIICase(b);
}

bool InspectorNodeFinder::checkStartsWith(const String& a, const String& b)
{
    if (m_caseSensitive)
        return a.startsWith(b);
    return a.startsWithIgnoringASCIICase(b);
}

bool InspectorNodeFinder::checkEndsWith(const String& a, const String& b)
{
    if (m_caseSensitive)
        return a.endsWith(b);
    return a.endsWithIgnoringASCIICase(b);
}

bool InspectorNodeFinder::matchesAttribute(const Attribute& attribute)
{
    if (checkContains(attribute.localName().string(), m_query))
        return true;

    auto value = attribute.value().string();
    return m_exactAttributeMatch ? checkEquals(value, m_attributeQuery) : checkContains(value, m_attributeQuery);
}

bool InspectorNodeFinder::matchesElement(const Element& element)
{
    String nodeName = element.nodeName();
    if ((!m_startTagFound && !m_endTagFound && checkContains(nodeName, m_tagNameQuery))
        || (m_startTagFound && m_endTagFound && checkEquals(nodeName, m_tagNameQuery))
        || (m_startTagFound && !m_endTagFound && checkStartsWith(nodeName, m_tagNameQuery))
        || (!m_startTagFound && m_endTagFound && checkEndsWith(nodeName, m_tagNameQuery)))
        return true;

    if (!element.hasAttributes())
        return false;

    for (auto& attribute : element.attributes()) {
        if (matchesAttribute(attribute))
            return true;
    }

    return false;
}

void InspectorNodeFinder::searchUsingXPath(Node& parentNode)
{
    auto evaluateResult = parentNode.document().evaluate(m_query, parentNode, nullptr, XPathResult::ORDERED_NODE_SNAPSHOT_TYPE, nullptr);
    if (evaluateResult.hasException())
        return;
    auto result = evaluateResult.releaseReturnValue();

    auto snapshotLengthResult = result->snapshotLength();
    if (snapshotLengthResult.hasException())
        return;
    unsigned size = snapshotLengthResult.releaseReturnValue();

    for (unsigned i = 0; i < size; ++i) {
        auto snapshotItemResult = result->snapshotItem(i);
        if (snapshotItemResult.hasException())
            return;
        Node* node = snapshotItemResult.releaseReturnValue();

        if (auto* attr = dynamicDowncast<Attr>(*node))
            node = attr->ownerElement();

        // XPath can get out of the context node that we pass as the starting point to evaluate, so we need to filter for just the nodes we care about.
        if (parentNode.contains(node))
            m_results.add(node);
    }
}

void InspectorNodeFinder::searchUsingCSSSelectors(Node& parentNode)
{
    RefPtr container = dynamicDowncast<ContainerNode>(parentNode);
    if (!container)
        return;

    auto queryResult = container->querySelectorAll(m_query);
    if (queryResult.hasException())
        return;

    auto nodeList = queryResult.releaseReturnValue();
    unsigned size = nodeList->length();
    for (unsigned i = 0; i < size; ++i)
        m_results.add(nodeList->item(i));
}

} // namespace WebCore
