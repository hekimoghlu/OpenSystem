/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#include "XPathResult.h"

#include "Document.h"
#include "XPathEvaluator.h"

namespace WebCore {

XPathResult::XPathResult(Document& document, const XPath::Value& value)
    : m_value(value)
{
    switch (m_value.type()) {
    case XPath::Value::Type::Boolean:
        m_resultType = BOOLEAN_TYPE;
        return;
    case XPath::Value::Type::Number:
        m_resultType = NUMBER_TYPE;
        return;
    case XPath::Value::Type::String:
        m_resultType = STRING_TYPE;
        return;
    case XPath::Value::Type::NodeSet:
        m_resultType = UNORDERED_NODE_ITERATOR_TYPE;
        m_nodeSetPosition = 0;
        m_nodeSet = m_value.toNodeSet();
        m_document = &document;
        m_domTreeVersion = document.domTreeVersion();
        return;
    }
    ASSERT_NOT_REACHED();
}

XPathResult::~XPathResult() = default;

ExceptionOr<void> XPathResult::convertTo(unsigned short type)
{
    switch (type) {
    case ANY_TYPE:
        break;
    case NUMBER_TYPE:
        m_resultType = type;
        m_value = m_value.toNumber();
        break;
    case STRING_TYPE:
        m_resultType = type;
        m_value = m_value.toString();
        break;
    case BOOLEAN_TYPE:
        m_resultType = type;
        m_value = m_value.toBoolean();
        break;
    case UNORDERED_NODE_ITERATOR_TYPE:
    case UNORDERED_NODE_SNAPSHOT_TYPE:
    case ANY_UNORDERED_NODE_TYPE:
    case FIRST_ORDERED_NODE_TYPE: // This is correct - singleNodeValue() will take care of ordering.
        if (!m_value.isNodeSet())
            return Exception { ExceptionCode::TypeError };
        m_resultType = type;
        break;
    case ORDERED_NODE_ITERATOR_TYPE:
        if (!m_value.isNodeSet())
            return Exception { ExceptionCode::TypeError };
        m_nodeSet.sort();
        m_resultType = type;
        break;
    case ORDERED_NODE_SNAPSHOT_TYPE:
        if (!m_value.isNodeSet())
            return Exception { ExceptionCode::TypeError };
        m_value.toNodeSet().sort();
        m_resultType = type;
        break;
    }
    return { };
}

unsigned short XPathResult::resultType() const
{
    return m_resultType;
}

ExceptionOr<double> XPathResult::numberValue() const
{
    if (resultType() != NUMBER_TYPE)
        return Exception { ExceptionCode::TypeError };
    return m_value.toNumber();
}

ExceptionOr<String> XPathResult::stringValue() const
{
    if (resultType() != STRING_TYPE)
        return Exception { ExceptionCode::TypeError };
    return m_value.toString();
}

ExceptionOr<bool> XPathResult::booleanValue() const
{
    if (resultType() != BOOLEAN_TYPE)
        return Exception { ExceptionCode::TypeError };
    return m_value.toBoolean();
}

ExceptionOr<Node*> XPathResult::singleNodeValue() const
{
    if (resultType() != ANY_UNORDERED_NODE_TYPE && resultType() != FIRST_ORDERED_NODE_TYPE)
        return Exception { ExceptionCode::TypeError };

    auto& nodes = m_value.toNodeSet();
    if (resultType() == FIRST_ORDERED_NODE_TYPE)
        return nodes.firstNode();
    else
        return nodes.anyNode();
}

bool XPathResult::invalidIteratorState() const
{
    if (resultType() != UNORDERED_NODE_ITERATOR_TYPE && resultType() != ORDERED_NODE_ITERATOR_TYPE)
        return false;

    ASSERT(m_document);
    return m_document->domTreeVersion() != m_domTreeVersion;
}

ExceptionOr<unsigned> XPathResult::snapshotLength() const
{
    if (resultType() != UNORDERED_NODE_SNAPSHOT_TYPE && resultType() != ORDERED_NODE_SNAPSHOT_TYPE)
        return Exception { ExceptionCode::TypeError };

    return m_value.toNodeSet().size();
}

ExceptionOr<Node*> XPathResult::iterateNext()
{
    if (resultType() != UNORDERED_NODE_ITERATOR_TYPE && resultType() != ORDERED_NODE_ITERATOR_TYPE)
        return Exception { ExceptionCode::TypeError };

    if (invalidIteratorState())
        return Exception { ExceptionCode::InvalidStateError };

    if (m_nodeSetPosition >= m_nodeSet.size())
        return nullptr;

    return m_nodeSet[m_nodeSetPosition++];
}

ExceptionOr<Node*> XPathResult::snapshotItem(unsigned index)
{
    if (resultType() != UNORDERED_NODE_SNAPSHOT_TYPE && resultType() != ORDERED_NODE_SNAPSHOT_TYPE)
        return Exception { ExceptionCode::TypeError };

    auto& nodes = m_value.toNodeSet();
    if (index >= nodes.size())
        return nullptr;

    return nodes[index];
}

} // namespace WebCore
