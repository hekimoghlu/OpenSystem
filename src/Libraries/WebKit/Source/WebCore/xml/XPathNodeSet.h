/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

namespace WebCore {
    namespace XPath {

        class NodeSet {
        public:
            NodeSet() : m_isSorted(true), m_subtreesAreDisjoint(false) { }
            explicit NodeSet(RefPtr<Node>&& node)
                : m_isSorted(true), m_subtreesAreDisjoint(false), m_nodes(1, WTFMove(node))
            { }
            
            size_t size() const { return m_nodes.size(); }
            bool isEmpty() const { return m_nodes.isEmpty(); }
            Node* operator[](unsigned i) const { return m_nodes.at(i).get(); }
            void reserveCapacity(size_t newCapacity) { m_nodes.reserveCapacity(newCapacity); }
            void clear() { m_nodes.clear(); }

            // NodeSet itself does not verify that nodes in it are unique.
            void append(RefPtr<Node>&& node) { m_nodes.append(WTFMove(node)); }
            void append(const NodeSet& nodeSet) { m_nodes.appendVector(nodeSet.m_nodes); }

            // Returns the set's first node in document order, or nullptr if the set is empty.
            Node* firstNode() const;

            // Returns nullptr if the set is empty.
            Node* anyNode() const;

            // NodeSet itself doesn't check if it contains nodes in document order - the caller should tell it if it does not.
            void markSorted(bool isSorted) { m_isSorted = isSorted; }
            bool isSorted() const { return m_isSorted || m_nodes.size() < 2; }

            void sort() const;

            // No node in the set is ancestor of another. Unlike m_isSorted, this is assumed to be false, unless the caller sets it to true.
            void markSubtreesDisjoint(bool disjoint) { m_subtreesAreDisjoint = disjoint; }
            bool subtreesAreDisjoint() const { return m_subtreesAreDisjoint || m_nodes.size() < 2; }

            const RefPtr<Node>* begin() const { return m_nodes.begin(); }
            const RefPtr<Node>* end() const { return m_nodes.end(); }

        private:
            void traversalSort() const;

            mutable bool m_isSorted;
            bool m_subtreesAreDisjoint;
            mutable Vector<RefPtr<Node>> m_nodes;
        };

    } // namespace XPath
} // namespace WebCore
