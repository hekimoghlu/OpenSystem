/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#include "DFGFrozenValue.h"
#include "GPRInfo.h"
#include <wtf/text/StringImpl.h>

namespace JSC {

class CCallHelpers;

namespace DFG {

class Graph;
class Plan;

// Represents either a JSValue, or for JSValues that require allocation in the heap,
// it tells you everything you'd need to know in order to allocate it.

class LazyJSValue {
public:
    enum LazinessKind {
        KnownValue,
        SingleCharacterString,
        KnownStringImpl,
        NewStringImpl
    };

    LazyJSValue(FrozenValue* value = FrozenValue::emptySingleton())
        : m_kind(KnownValue)
    {
        u.value = value;
    }
    
    static LazyJSValue singleCharacterString(UChar character)
    {
        LazyJSValue result;
        result.m_kind = SingleCharacterString;
        result.u.character = character;
        return result;
    }
    
    static LazyJSValue knownStringImpl(AtomStringImpl* string)
    {
        LazyJSValue result;
        result.m_kind = KnownStringImpl;
        result.u.stringImpl = string;
        return result;
    }

    static LazyJSValue newString(Graph&, const String&);

    LazinessKind kind() const { return m_kind; }
    SpeculatedType speculatedType() const { return kind() == KnownValue ? SpecBytecodeTop : SpecString; }
    
    FrozenValue* tryGetValue(Graph&) const
    {
        if (m_kind == KnownValue)
            return value();
        return nullptr;
    }
    
    JSValue getValue(VM&) const;
    
    FrozenValue* value() const
    {
        ASSERT(m_kind == KnownValue);
        return u.value;
    }
    
    UChar character() const
    {
        ASSERT(m_kind == SingleCharacterString);
        return u.character;
    }

    String tryGetString(Graph&) const;
    
    StringImpl* stringImpl() const
    {
        ASSERT(m_kind == KnownStringImpl || m_kind == NewStringImpl);
        return u.stringImpl;
    }

    TriState strictEqual(const LazyJSValue& other) const;
    
    uintptr_t switchLookupValue(SwitchKind) const;

    void emit(CCallHelpers&, JSValueRegs, Plan&) const;
    
    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;
    
private:
    const StringImpl* tryGetStringImpl() const;
    
    union {
        FrozenValue* value;
        UChar character;
        StringImpl* stringImpl;
    } u;
    LazinessKind m_kind;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
