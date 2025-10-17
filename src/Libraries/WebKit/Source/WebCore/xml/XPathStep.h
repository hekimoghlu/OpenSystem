/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class Node;

namespace XPath {

class Expression;
class NodeSet;

class Step {
    WTF_MAKE_TZONE_ALLOCATED(Step);
public:
    enum Axis {
        AncestorAxis, AncestorOrSelfAxis, AttributeAxis,
        ChildAxis, DescendantAxis, DescendantOrSelfAxis,
        FollowingAxis, FollowingSiblingAxis, NamespaceAxis,
        ParentAxis, PrecedingAxis, PrecedingSiblingAxis,
        SelfAxis
    };

    class NodeTest {
        WTF_MAKE_TZONE_ALLOCATED(NodeTest);
    public:
        enum Kind { TextNodeTest, CommentNodeTest, ProcessingInstructionNodeTest, AnyNodeTest, NameTest };

        explicit NodeTest(Kind kind) : m_kind(kind) { }
        NodeTest(Kind kind, const AtomString& data) : m_kind(kind), m_data(data) { }
        NodeTest(Kind kind, const AtomString& data, const AtomString& namespaceURI) : m_kind(kind), m_data(data), m_namespaceURI(namespaceURI) { }

    private:
        friend class Step;
        friend void optimizeStepPair(Step&, Step&, bool&);
        friend bool nodeMatchesBasicTest(Node&, Axis, const NodeTest&);
        friend bool nodeMatches(Node&, Axis, const NodeTest&);

        Kind m_kind;
        AtomString m_data;
        AtomString m_namespaceURI;
        Vector<std::unique_ptr<Expression>> m_mergedPredicates;
    };

    Step(Axis, NodeTest);
    Step(Axis, NodeTest, Vector<std::unique_ptr<Expression>>);
    ~Step();

    void optimize();

    void evaluate(Node& context, NodeSet&) const;

    Axis axis() const { return m_axis; }

private:
    friend void optimizeStepPair(Step&, Step&, bool&);

    bool predicatesAreContextListInsensitive() const;

    void parseNodeTest(const String&);
    void nodesInAxis(Node& context, NodeSet&) const;

    Axis m_axis;
    NodeTest m_nodeTest;
    Vector<std::unique_ptr<Expression>> m_predicates;
};

void optimizeStepPair(Step&, Step&, bool& dropSecondStep);

} // namespace XPath
} // namespace WebCore
