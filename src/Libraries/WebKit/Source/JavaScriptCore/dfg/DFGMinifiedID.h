/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#include "DFGCommon.h"
#include <wtf/HashMap.h>
#include <wtf/Packed.h>
#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

class Graph;
class MinifiedNode;
class ValueSource;
struct Node;

class MinifiedID {
public:
    MinifiedID() = default;
    MinifiedID(WTF::HashTableDeletedValueType) : m_index(otherInvalidIndex()) { }
    explicit MinifiedID(Node* node);
    
    bool operator!() const { return m_index.get() == invalidIndex(); }
    
    bool operator==(const MinifiedID& other) const { return m_index.get() == other.m_index.get(); }
    bool operator<(const MinifiedID& other) const { return m_index.get() < other.m_index.get(); }
    bool operator>(const MinifiedID& other) const { return m_index.get() > other.m_index.get(); }
    bool operator<=(const MinifiedID& other) const { return m_index.get() <= other.m_index.get(); }
    bool operator>=(const MinifiedID& other) const { return m_index.get() >= other.m_index.get(); }
    
    unsigned hash() const { return WTF::IntHash<unsigned>::hash(m_index.get()); }
    
    void dump(PrintStream& out) const { out.print(m_index.get()); }
    
    bool isHashTableDeletedValue() const { return m_index.get() == otherInvalidIndex(); }
    
    static MinifiedID fromBits(unsigned value)
    {
        MinifiedID result;
        result.m_index = value;
        return result;
    }
    
    unsigned bits() const { return m_index.get(); }

private:
    friend class MinifiedNode;
    
    static constexpr unsigned invalidIndex() { return static_cast<unsigned>(-1); }
    static constexpr unsigned otherInvalidIndex() { return static_cast<unsigned>(-2); }
    
    Packed<unsigned> m_index { invalidIndex() };
};

struct MinifiedIDHash {
    static unsigned hash(const MinifiedID& key) { return key.hash(); }
    static bool equal(const MinifiedID& a, const MinifiedID& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::DFG

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::DFG::MinifiedID> : JSC::DFG::MinifiedIDHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::DFG::MinifiedID> : SimpleClassHashTraits<JSC::DFG::MinifiedID> {
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF
