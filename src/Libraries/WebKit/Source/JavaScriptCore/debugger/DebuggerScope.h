/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

#include "DebuggerLocation.h"
#include "JSObject.h"

namespace JSC {

class DebuggerCallFrame;
class JSScope;

class DebuggerScope final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesPut;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.debuggerScopeSpace<mode>();
    }

    JS_EXPORT_PRIVATE static DebuggerScope* create(VM& vm, JSScope* scope);

    DECLARE_VISIT_CHILDREN;
    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);

    DECLARE_EXPORT_INFO;

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject) 
    {
        return Structure::create(vm, globalObject, jsNull(), TypeInfo(ObjectType, StructureFlags), info()); 
    } 
    class iterator {
    public:
        iterator(DebuggerScope* node)
            : m_node(node)
        {
        }

        DebuggerScope* get() { return m_node; }
        iterator& operator++() { m_node = m_node->next(); return *this; }
        // postfix ++ intentionally omitted

        friend bool operator==(const iterator&, const iterator&) = default;

    private:
        DebuggerScope* m_node;
    };

    iterator begin();
    iterator end();
    DebuggerScope* next();

    void invalidateChain();
    bool isValid() const { return !!m_scope; }

    bool isCatchScope() const;
    bool isFunctionNameScope() const;
    bool isWithScope() const;
    bool isGlobalScope() const;
    bool isClosureScope() const;
    bool isGlobalLexicalEnvironment() const;
    bool isNestedLexicalScope() const;

    String name() const;
    DebuggerLocation location() const;

    JSValue caughtValue(JSGlobalObject*) const;

private:
    DebuggerScope(VM&, Structure*, JSScope*);
    DECLARE_DEFAULT_FINISH_CREATION;

    JSScope* jsScope() const { return m_scope.get(); }

    WriteBarrier<JSScope> m_scope;
    WriteBarrier<DebuggerScope> m_next;

    friend class DebuggerCallFrame;
};

inline DebuggerScope::iterator DebuggerScope::begin()
{
    return iterator(this); 
}

inline DebuggerScope::iterator DebuggerScope::end()
{ 
    return iterator(nullptr); 
}

} // namespace JSC
