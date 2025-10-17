/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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

#include <wtf/HashTraits.h>

namespace JSC {

class FunctionExecutable;
class JSObject;

struct ClassInfo;

class PrototypeKey {
public:
    PrototypeKey() { }
    
    PrototypeKey(JSObject* prototype, FunctionExecutable* executable, unsigned inlineCapacity, const ClassInfo* classInfo)
        : m_prototype(prototype)
        , m_executable(executable)
        , m_inlineCapacity(inlineCapacity)
        , m_classInfo(classInfo)
    {
    }
    
    PrototypeKey(WTF::HashTableDeletedValueType)
        : m_inlineCapacity(1)
    {
    }
    
    JSObject* prototype() const { return m_prototype; }
    FunctionExecutable* executable() const { return m_executable; }
    unsigned inlineCapacity() const { return m_inlineCapacity; }
    const ClassInfo* classInfo() const { return m_classInfo; }
    
    friend bool operator==(const PrototypeKey&, const PrototypeKey&) = default;
    
    explicit operator bool() const { return *this != PrototypeKey(); }
    bool isHashTableDeletedValue() const { return *this == PrototypeKey(WTF::HashTableDeletedValue); }
    
    unsigned hash() const
    {
        return WTF::IntHash<uintptr_t>::hash(std::bit_cast<uintptr_t>(m_prototype) ^ std::bit_cast<uintptr_t>(m_executable) ^ std::bit_cast<uintptr_t>(m_classInfo)) + m_inlineCapacity;
    }
    
private:
    // WARNING: We require all of these default values to be zero. Otherwise, you'll need to add
    // "static constexpr bool emptyValueIsZero = false;" to the HashTraits at the bottom of this file.
    JSObject* m_prototype { nullptr }; 
    FunctionExecutable* m_executable { nullptr }; 
    unsigned m_inlineCapacity { 0 };
    const ClassInfo* m_classInfo { nullptr };
};

struct PrototypeKeyHash {
    static unsigned hash(const PrototypeKey& key) { return key.hash(); }
    static bool equal(const PrototypeKey& a, const PrototypeKey& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

template<typename> struct DefaultHash;
template<> struct DefaultHash<JSC::PrototypeKey> : JSC::PrototypeKeyHash { };

template<typename> struct HashTraits;
template<> struct HashTraits<JSC::PrototypeKey> : SimpleClassHashTraits<JSC::PrototypeKey> { };

} // namespace WTF
