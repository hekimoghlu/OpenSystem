/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

#include "Node.h"
#include "Position.h"

namespace WebCore {

// A Position iterator with constant-time
// increment, decrement, and several predicates on the Position it is at.
// Conversion to/from Position is O(n) in the offset.
class PositionIterator {
public:
    PositionIterator(const Position& pos)
        : m_anchorNode(pos.anchorNode())
        , m_nodeAfterPositionInAnchor(m_anchorNode->traverseToChildAt(pos.deprecatedEditingOffset()))
        , m_offsetInAnchor(m_nodeAfterPositionInAnchor ? 0 : pos.deprecatedEditingOffset())
    {
    }
    operator Position() const;

    void increment();
    void decrement();

    Node* node() const { return m_anchorNode.get(); }
    RefPtr<Node> protectedNode() const { return m_anchorNode; }
    int offsetInLeafNode() const { return m_offsetInAnchor; }

    bool atStart() const;
    bool atEnd() const;
    bool atStartOfNode() const;
    bool atEndOfNode() const;
    bool isCandidate() const;

private:
    RefPtr<Node> m_anchorNode;
    RefPtr<Node> m_nodeAfterPositionInAnchor; // If this is non-null, m_nodeAfterPositionInAnchor->parentNode() == m_anchorNode;
    int m_offsetInAnchor { 0 };
};

} // namespace WebCore
