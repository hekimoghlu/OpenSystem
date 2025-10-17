/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#include "DumpContext.h"
#include <wtf/HashMap.h>
#include <wtf/PrintStream.h>

namespace JSC {

class JSGlobalObject;

namespace DFG {

class DesiredGlobalProperty {
public:
    DesiredGlobalProperty() = default;

    DesiredGlobalProperty(JSGlobalObject* globalObject, unsigned identifierNumber)
        : m_globalObject(globalObject)
        , m_identifierNumber(identifierNumber)
    {
    }

    DesiredGlobalProperty(WTF::HashTableDeletedValueType)
        : m_globalObject(nullptr)
        , m_identifierNumber(UINT_MAX)
    {
    }

    JSGlobalObject* globalObject() const { return m_globalObject; }
    unsigned identifierNumber() const { return m_identifierNumber; }

    friend bool operator==(const DesiredGlobalProperty&, const DesiredGlobalProperty&) = default;

    bool isHashTableDeletedValue() const
    {
        return !m_globalObject && m_identifierNumber == UINT_MAX;
    }

    unsigned hash() const
    {
        return WTF::PtrHash<JSGlobalObject*>::hash(m_globalObject) + m_identifierNumber * 7;
    }

    void dumpInContext(PrintStream& out, DumpContext*) const
    {
        out.print(m_identifierNumber, " for ", RawPointer(m_globalObject));
    }

    void dump(PrintStream& out) const
    {
        dumpInContext(out, nullptr);
    }

private:
    JSGlobalObject* m_globalObject { nullptr };
    unsigned m_identifierNumber { 0 };
};

struct DesiredGlobalPropertyHash {
    static unsigned hash(const DesiredGlobalProperty& key) { return key.hash(); }
    static bool equal(const DesiredGlobalProperty& a, const DesiredGlobalProperty& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} } // namespace JSC::DFG

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::DFG::DesiredGlobalProperty> : JSC::DFG::DesiredGlobalPropertyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::DFG::DesiredGlobalProperty> : SimpleClassHashTraits<JSC::DFG::DesiredGlobalProperty> { };

} // namespace WTF

#endif // ENABLE(DFG_JIT)
