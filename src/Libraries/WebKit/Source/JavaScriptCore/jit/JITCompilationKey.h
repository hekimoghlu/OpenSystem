/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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

#include "JITCompilationMode.h"
#include <wtf/HashMap.h>

namespace JSC {

class JSCell;

class JITCompilationKey {
public:
    JITCompilationKey()
        : m_codeBlock(nullptr)
        , m_mode(JITCompilationMode::InvalidCompilation)
    {
    }
    
    JITCompilationKey(WTF::HashTableDeletedValueType)
        : m_codeBlock(nullptr)
        , m_mode(JITCompilationMode::DFG)
    {
    }
    
    JITCompilationKey(JSCell* profiledBlock, JITCompilationMode mode)
        : m_codeBlock(profiledBlock)
        , m_mode(mode)
    {
    }

    explicit operator bool() const
    {
        return m_codeBlock || m_mode != JITCompilationMode::InvalidCompilation;
    }
    
    bool isHashTableDeletedValue() const
    {
        return !m_codeBlock && m_mode != JITCompilationMode::InvalidCompilation;
    }
    
    JITCompilationMode mode() const { return m_mode; }
    
    friend bool operator==(const JITCompilationKey&, const JITCompilationKey&) = default;
    
    unsigned hash() const
    {
        return WTF::pairIntHash(WTF::PtrHash<JSCell*>::hash(m_codeBlock), static_cast<std::underlying_type<JITCompilationMode>::type>(m_mode));
    }
    
    void dump(PrintStream&) const;

private:
    // Either CodeBlock* or UnlinkedCodeBlock* for basleline JIT.
    JSCell* m_codeBlock;
    JITCompilationMode m_mode;
};

struct JITCompilationKeyHash {
    static unsigned hash(const JITCompilationKey& key) { return key.hash(); }
    static bool equal(const JITCompilationKey& a, const JITCompilationKey& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::JITCompilationKey> : JSC::JITCompilationKeyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::JITCompilationKey> : SimpleClassHashTraits<JSC::JITCompilationKey> { };

} // namespace WTF
