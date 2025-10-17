/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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

#include "ConcurrentJSLock.h"
#include "Operands.h"
#include "ValueProfile.h"
#include "VirtualRegister.h"
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/SegmentedVector.h>

namespace JSC {

class ScriptExecutable;

class LazyOperandValueProfileKey {
public:
    LazyOperandValueProfileKey()
        : m_operand(VirtualRegister()) // not a valid operand index in our current scheme
    {
    }
    
    LazyOperandValueProfileKey(WTF::HashTableDeletedValueType)
        : m_bytecodeIndex(WTF::HashTableDeletedValue)
        , m_operand(VirtualRegister()) // not a valid operand index in our current scheme
    {
    }
    
    LazyOperandValueProfileKey(BytecodeIndex bytecodeIndex, Operand operand)
        : m_bytecodeIndex(bytecodeIndex)
        , m_operand(operand)
    {
        ASSERT(m_operand.isValid());
    }
    
    bool operator!() const
    {
        return !m_operand.isValid();
    }
    
    friend bool operator==(const LazyOperandValueProfileKey&, const LazyOperandValueProfileKey&) = default;
    
    unsigned hash() const
    {
        return m_bytecodeIndex.hash() + m_operand.value() + static_cast<unsigned>(m_operand.kind());
    }
    
    BytecodeIndex bytecodeIndex() const
    {
        ASSERT(!!*this);
        return m_bytecodeIndex;
    }

    Operand operand() const
    {
        ASSERT(!!*this);
        return m_operand;
    }
    
    bool isHashTableDeletedValue() const
    {
        return !m_operand.isValid() && m_bytecodeIndex.isHashTableDeletedValue();
    }
private: 
    BytecodeIndex m_bytecodeIndex;
    Operand m_operand;
};

struct LazyOperandValueProfileKeyHash {
    static unsigned hash(const LazyOperandValueProfileKey& key) { return key.hash(); }
    static bool equal(
        const LazyOperandValueProfileKey& a,
        const LazyOperandValueProfileKey& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::LazyOperandValueProfileKey> : JSC::LazyOperandValueProfileKeyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::LazyOperandValueProfileKey> : public GenericHashTraits<JSC::LazyOperandValueProfileKey> {
    static void constructDeletedValue(JSC::LazyOperandValueProfileKey& slot) { new (NotNull, &slot) JSC::LazyOperandValueProfileKey(HashTableDeletedValue); }
    static bool isDeletedValue(const JSC::LazyOperandValueProfileKey& value) { return value.isHashTableDeletedValue(); }
};

} // namespace WTF

namespace JSC {

struct LazyOperandValueProfile : public MinimalValueProfile {
    LazyOperandValueProfile()
        : MinimalValueProfile()
        , m_operand(VirtualRegister())
    {
    }
    
    explicit LazyOperandValueProfile(const LazyOperandValueProfileKey& key)
        : MinimalValueProfile()
        , m_key(key)
    {
    }
    
    LazyOperandValueProfileKey key() const
    {
        return m_key;
    }

    VirtualRegister m_operand;
    LazyOperandValueProfileKey m_key;
};

} // namespace JSC
