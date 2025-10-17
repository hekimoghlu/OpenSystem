/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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

#include <wtf/ListHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Attribute;
class Element;
class Node;

class InspectorNodeFinder {
public:
    InspectorNodeFinder(const String& query, bool caseSensitive);
    void performSearch(Node*);
    const ListHashSet<Node*>& results() const { return m_results; }

private:
    bool checkEquals(const String&, const String&);
    bool checkContains(const String&, const String&);
    bool checkStartsWith(const String&, const String&);
    bool checkEndsWith(const String&, const String&);

    bool matchesAttribute(const Attribute&);
    bool matchesElement(const Element&);

    void searchUsingDOMTreeTraversal(Node&);
    void searchUsingXPath(Node&);
    void searchUsingCSSSelectors(Node&);

    String m_query;
    String m_tagNameQuery;
    String m_attributeQuery;
    bool m_caseSensitive;

    ListHashSet<Node*> m_results;
    bool m_startTagFound;
    bool m_endTagFound;
    bool m_exactAttributeMatch;
};

} // namespace WebCore
