/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

#include "ExceptionOr.h"
#include "XPathValue.h"

namespace WebCore {

class XPathResult : public RefCounted<XPathResult> {
public:
    enum XPathResultType {
        ANY_TYPE = 0,
        NUMBER_TYPE = 1,
        STRING_TYPE = 2,
        BOOLEAN_TYPE = 3,
        UNORDERED_NODE_ITERATOR_TYPE = 4,
        ORDERED_NODE_ITERATOR_TYPE = 5,
        UNORDERED_NODE_SNAPSHOT_TYPE = 6,
        ORDERED_NODE_SNAPSHOT_TYPE = 7,
        ANY_UNORDERED_NODE_TYPE = 8,
        FIRST_ORDERED_NODE_TYPE = 9
    };

    static Ref<XPathResult> create(Document& document, const XPath::Value& value) { return adoptRef(*new XPathResult(document, value)); }
    WEBCORE_EXPORT ~XPathResult();

    ExceptionOr<void> convertTo(unsigned short type);

    WEBCORE_EXPORT unsigned short resultType() const;

    WEBCORE_EXPORT ExceptionOr<double> numberValue() const;
    WEBCORE_EXPORT ExceptionOr<String> stringValue() const;
    WEBCORE_EXPORT ExceptionOr<bool> booleanValue() const;
    WEBCORE_EXPORT ExceptionOr<Node*> singleNodeValue() const;

    WEBCORE_EXPORT bool invalidIteratorState() const;
    WEBCORE_EXPORT ExceptionOr<unsigned> snapshotLength() const;
    WEBCORE_EXPORT ExceptionOr<Node*> iterateNext();
    WEBCORE_EXPORT ExceptionOr<Node*> snapshotItem(unsigned index);

    const XPath::Value& value() const { return m_value; }

private:
    XPathResult(Document&, const XPath::Value&);

    XPath::Value m_value;
    unsigned m_nodeSetPosition { 0 };
    XPath::NodeSet m_nodeSet; // FIXME: why duplicate the node set stored in m_value?
    unsigned short m_resultType;
    RefPtr<Document> m_document;
    uint64_t m_domTreeVersion { 0 };
};

} // namespace WebCore
