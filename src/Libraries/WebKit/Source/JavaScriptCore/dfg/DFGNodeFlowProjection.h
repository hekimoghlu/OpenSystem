/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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

#if ENABLE(DFG_JIT)

#include "DFGNode.h"
#include <wtf/HashTable.h>

namespace JSC { namespace DFG {

class NodeFlowProjection {
public:
    enum Kind {
        Primary,
        Shadow
    };
    
    NodeFlowProjection() { }
    
    NodeFlowProjection(Node* node)
        : m_word(std::bit_cast<uintptr_t>(node))
    {
        ASSERT(kind() == Primary);
    }
    
    NodeFlowProjection(Node* node, Kind kind)
        : m_word(std::bit_cast<uintptr_t>(node) | (kind == Shadow ? shadowBit : 0))
    {
        ASSERT(this->kind() == kind);
    }
    
    NodeFlowProjection(WTF::HashTableDeletedValueType)
        : m_word(shadowBit)
    {
    }
    
    explicit operator bool() const { return !!m_word; }
    
    Kind kind() const { return (m_word & shadowBit) ? Shadow : Primary; }
    
    Node* node() const { return std::bit_cast<Node*>(m_word & ~shadowBit); }
    
    Node& operator*() const { return *node(); }
    Node* operator->() const { return node(); }
    
    unsigned hash() const
    {
        return m_word;
    }
    
    friend bool operator==(const NodeFlowProjection&, const NodeFlowProjection&) = default;
    
    bool operator<(NodeFlowProjection other) const
    {
        if (kind() != other.kind())
            return kind() < other.kind();
        return node() < other.node();
    }
    
    bool operator>(NodeFlowProjection other) const
    {
        return other < *this;
    }
    
    bool operator<=(NodeFlowProjection other) const
    {
        return !(*this > other);
    }
    
    bool operator>=(NodeFlowProjection other) const
    {
        return !(*this < other);
    }
    
    bool isHashTableDeletedValue() const
    {
        return *this == NodeFlowProjection(WTF::HashTableDeletedValue);
    }
    
    // Phi shadow projections can become invalid because the Phi might be folded to something else.
    bool isStillValid() const
    {
        return *this && (kind() == Primary || node()->op() == Phi);
    }
    
    void dump(PrintStream&) const;
    
    template<typename Func>
    static void forEach(Node* node, const Func& func)
    {
        func(NodeFlowProjection(node));
        if (node->op() == Phi)
            func(NodeFlowProjection(node, Shadow));
    }
    
public:
    static constexpr uintptr_t shadowBit = 1;
    
    uintptr_t m_word { 0 };
};

struct NodeFlowProjectionHash {
    static unsigned hash(NodeFlowProjection key) { return key.hash(); }
    static bool equal(NodeFlowProjection a, NodeFlowProjection b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::DFG

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::DFG::NodeFlowProjection> : JSC::DFG::NodeFlowProjectionHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::DFG::NodeFlowProjection> : SimpleClassHashTraits<JSC::DFG::NodeFlowProjection> { };

} // namespace WTF

#endif // ENABLE(DFG_JIT)

