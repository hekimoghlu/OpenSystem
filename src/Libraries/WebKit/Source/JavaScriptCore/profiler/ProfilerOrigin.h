/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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

#include "BytecodeIndex.h"
#include "CodeBlockHash.h"
#include "JSCJSValue.h"
#include <wtf/PrintStream.h>

namespace JSC {

class CodeBlock;

namespace Profiler {

class Bytecodes;
class Database;
class Dumper;

class Origin {
public:
    Origin() = default;
    Origin(WTF::HashTableDeletedValueType)
        : m_bytecodeIndex(WTF::HashTableDeletedValue)
    { }
    
    Origin(Bytecodes* bytecodes, BytecodeIndex bytecodeIndex)
        : m_bytecodes(bytecodes)
        , m_bytecodeIndex(bytecodeIndex)
    {
        ASSERT(m_bytecodeIndex.offset() < std::numeric_limits<unsigned>::max() - 1);
    }
    
    Origin(Database&, CodeBlock*, BytecodeIndex);
    
    bool operator!() const { return !m_bytecodeIndex; }
    
    Bytecodes* bytecodes() const { return m_bytecodes; }
    BytecodeIndex bytecodeIndex() const { return m_bytecodeIndex; }

    friend bool operator==(const Origin&, const Origin&) = default;
    unsigned hash() const;
    
    bool isHashTableDeletedValue() const;
    
    void dump(PrintStream&) const;
    Ref<JSON::Value> toJSON(Dumper&) const;

private:
    Bytecodes* m_bytecodes;
    BytecodeIndex m_bytecodeIndex;
};

inline unsigned Origin::hash() const
{
    return WTF::PtrHash<Bytecodes*>::hash(m_bytecodes) + m_bytecodeIndex.hash();
}

inline bool Origin::isHashTableDeletedValue() const
{
    return m_bytecodeIndex.isHashTableDeletedValue();
}

struct OriginHash {
    static unsigned hash(const Origin& key) { return key.hash(); }
    static bool equal(const Origin& a, const Origin& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::Profiler

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::Profiler::Origin> : JSC::Profiler::OriginHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::Profiler::Origin> : SimpleClassHashTraits<JSC::Profiler::Origin> { };

} // namespace WTF
