/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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

#include "DFGCommon.h"
#include "DFGUseKind.h"

namespace JSC { namespace DFG {

class AdjacencyList;

class Edge {
public:
    explicit Edge(Node* node = nullptr, UseKind useKind = UntypedUse, ProofStatus proofStatus = NeedsCheck, KillStatus killStatus = DoesNotKill)
#if USE(JSVALUE64)
        : m_encodedWord(makeWord(node, useKind, proofStatus, killStatus))
#else
        : m_node(node)
        , m_encodedWord(makeWord(useKind, proofStatus, killStatus))
#endif
    {
    }
    
#if USE(JSVALUE64)
    static constexpr uintptr_t shift() { return 48; }
    static constexpr uintptr_t addressMask() { return ~(0xffull << shift()); }

    Node* node() const { return std::bit_cast<Node*>(m_encodedWord & addressMask()); }
#else
    Node* node() const { return m_node; }
#endif

    Node& operator*() const { return *node(); }
    Node* operator->() const { return node(); }
    
    void setNode(Node* node)
    {
#if USE(JSVALUE64)
        m_encodedWord = makeWord(node, useKind(), proofStatus(), killStatus());
#else
        m_node = node;
#endif
    }
    
    UseKind useKindUnchecked() const
    {
#if USE(JSVALUE64)
        unsigned shifted = m_encodedWord >> (2 + shift()) & 0x3f;
#else
        unsigned shifted = static_cast<UseKind>(m_encodedWord) >> 2;
#endif
        ASSERT(shifted < static_cast<unsigned>(LastUseKind));
        UseKind result = static_cast<UseKind>(shifted);
        ASSERT(node() || result == UntypedUse);
        return result;
    }
    UseKind useKind() const
    {
        ASSERT(node());
        return useKindUnchecked();
    }
    void setUseKind(UseKind useKind)
    {
        ASSERT(node());
#if USE(JSVALUE64)
        m_encodedWord = makeWord(node(), useKind, proofStatus(), killStatus());
#else
        m_encodedWord = makeWord(useKind, proofStatus(), killStatus());
#endif
    }
    
    ProofStatus proofStatusUnchecked() const
    {
#if USE(JSVALUE64)
        return proofStatusForIsProved(m_encodedWord >> shift() & 1);
#else
        return proofStatusForIsProved(m_encodedWord & 1);
#endif
    }
    ProofStatus proofStatus() const
    {
        ASSERT(node());
        return proofStatusUnchecked();
    }
    void setProofStatus(ProofStatus proofStatus)
    {
        ASSERT(node());
#if USE(JSVALUE64)
        m_encodedWord = makeWord(node(), useKind(), proofStatus, killStatus());
#else
        m_encodedWord = makeWord(useKind(), proofStatus, killStatus());
#endif
    }
    bool isProved() const
    {
        return proofStatus() == IsProved;
    }
    
    bool willNotHaveCheck() const
    {
        return isProved() || shouldNotHaveTypeCheck(useKind());
    }
    bool willHaveCheck() const
    {
        return !willNotHaveCheck();
    }
    
    KillStatus killStatusUnchecked() const
    {
#if USE(JSVALUE64)
        return killStatusForDoesKill(m_encodedWord >> shift() & 2);
#else
        return killStatusForDoesKill(m_encodedWord & 2);
#endif
    }
    KillStatus killStatus() const
    {
        ASSERT(node());
        return killStatusUnchecked();
    }
    void setKillStatus(KillStatus killStatus)
    {
        ASSERT(node());
#if USE(JSVALUE64)
        m_encodedWord = makeWord(node(), useKind(), proofStatus(), killStatus);
#else
        m_encodedWord = makeWord(useKind(), proofStatus(), killStatus);
#endif
    }
    bool doesKill() const { return DFG::doesKill(killStatus()); }

    bool isSet() const { return !!node(); }

    Edge sanitized() const
    {
        Edge result = *this;
#if USE(JSVALUE64)
        result.m_encodedWord = makeWord(node(), useKindUnchecked(), NeedsCheck, DoesNotKill);
#else
        result.m_encodedWord = makeWord(useKindUnchecked(), NeedsCheck, DoesNotKill);
#endif
        return result;
    }

    bool operator!() const { return !isSet(); }
    explicit operator bool() const { return isSet(); }
    
    friend bool operator==(const Edge&, const Edge&) = default;

    void dump(PrintStream&) const;
    
    unsigned hash() const
    {
#if USE(JSVALUE64)
        return IntHash<uintptr_t>::hash(m_encodedWord);
#else
        return PtrHash<Node*>::hash(m_node) + m_encodedWord;
#endif
    }

private:
    friend class AdjacencyList;
    
#if USE(JSVALUE64)
    static uintptr_t makeWord(Node* node, UseKind useKind, ProofStatus proofStatus, KillStatus killStatus)
    {
        ASSERT(sizeof(node) == 8);
        uintptr_t maskedPointer = std::bit_cast<uintptr_t>(node) & addressMask();
        ASSERT(maskedPointer == std::bit_cast<uintptr_t>(node));
        ASSERT(useKind < LastUseKind);
        static_assert((static_cast<uintptr_t>(LastUseKind) << 2) < (static_cast<uintptr_t>(1) << shift()), "We rely on this being true to not clobber the node pointer.");
        uintptr_t kindBits = (static_cast<uintptr_t>(useKind) << 2) | (DFG::doesKill(killStatus) << 1) | static_cast<uintptr_t>(DFG::isProved(proofStatus));
        kindBits <<= shift();
        uintptr_t result = maskedPointer | kindBits;

        if (ASSERT_ENABLED) {
            union U {
                U() { word = 0; }
                uintptr_t word;
                Edge edge;
            } u;
            u.word = result;
            ASSERT(u.edge.useKindUnchecked() == useKind);
            ASSERT(u.edge.node() == node);
            ASSERT(u.edge.proofStatusUnchecked() == proofStatus);
            ASSERT(u.edge.killStatusUnchecked() == killStatus);
        }
        return result;
    }
    
#else
    static uintptr_t makeWord(UseKind useKind, ProofStatus proofStatus, KillStatus killStatus)
    {
        return (static_cast<uintptr_t>(useKind) << 2) | (DFG::doesKill(killStatus) << 1) | static_cast<uintptr_t>(DFG::isProved(proofStatus));
    }
    
    Node* m_node;
#endif
    // On 64-bit this holds both the pointer and the use kind, while on 32-bit
    // this just holds the use kind. In both cases this may be hijacked by
    // AdjacencyList for storing firstChild and numChildren.
    uintptr_t m_encodedWord;
};

inline bool operator==(Edge edge, Node* node)
{
    return edge.node() == node;
}
inline bool operator==(Node* node, Edge edge)
{
    return edge.node() == node;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
