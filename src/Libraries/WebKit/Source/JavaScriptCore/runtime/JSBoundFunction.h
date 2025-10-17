/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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

#include "JSFunction.h"
#include "JSImmutableButterfly.h"

namespace JSC {

JSC_DECLARE_HOST_FUNCTION(boundThisNoArgsFunctionCall);
JSC_DECLARE_HOST_FUNCTION(boundFunctionCall);
JSC_DECLARE_HOST_FUNCTION(boundFunctionConstruct);
JSC_DECLARE_HOST_FUNCTION(isBoundFunction);
JSC_DECLARE_HOST_FUNCTION(hasInstanceBoundFunction);

class JSBoundFunction final : public JSFunction {
public:
    using Base = JSFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags & ~ImplementsDefaultHasInstance;
    static_assert(StructureFlags & ImplementsHasInstance);
    static constexpr unsigned maxEmbeddedArgs = 3; // Keep sizeof(JSBoundFunction) <= 96.

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.boundFunctionSpace<mode>();
    }

    JS_EXPORT_PRIVATE static JSBoundFunction* create(VM&, JSGlobalObject*, JSObject* targetFunction, JSValue boundThis, ArgList, double length, JSString* nameMayBeNull);
    static JSBoundFunction* createRaw(VM&, JSGlobalObject*, JSFunction* targetFunction, unsigned boundArgsLength, JSValue boundThis, JSValue arg0, JSValue arg1, JSValue arg2);
    
    static bool customHasInstance(JSObject*, JSGlobalObject*, JSValue);

    JSObject* targetFunction() { return m_targetFunction.get(); }
    JSValue boundThis() { return m_boundThis.get(); }
    unsigned boundArgsLength() const { return m_boundArgsLength; }
    JSArray* boundArgsCopy(JSGlobalObject*);
    JSString* nameMayBeNull() { return m_nameMayBeNull.get(); }
    JSString* name()
    {
        if (m_nameMayBeNull)
            return m_nameMayBeNull.get();
        return nameSlow(vm());
    }
    String nameString()
    {
        if (!m_nameMayBeNull)
            name();
        ASSERT(!m_nameMayBeNull->isRope());
        bool allocationAllowed = false;
        return m_nameMayBeNull->tryGetValue(allocationAllowed);
    }
    String nameStringWithoutGC(VM& vm)
    {
        if (m_nameMayBeNull) {
            ASSERT(!m_nameMayBeNull->isRope());
            bool allocationAllowed = false;
            return m_nameMayBeNull->tryGetValue(allocationAllowed);
        }
        return nameStringWithoutGCSlow(vm);
    }

    double length(VM& vm)
    {
        if (std::isnan(m_length))
            return lengthSlow(vm);
        return m_length;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
    
    static constexpr ptrdiff_t offsetOfTargetFunction() { return OBJECT_OFFSETOF(JSBoundFunction, m_targetFunction); }
    static constexpr ptrdiff_t offsetOfBoundThis() { return OBJECT_OFFSETOF(JSBoundFunction, m_boundThis); }
    static constexpr ptrdiff_t offsetOfBoundArgs() { return OBJECT_OFFSETOF(JSBoundFunction, m_boundArgs); }
    static constexpr ptrdiff_t offsetOfBoundArgsLength() { return OBJECT_OFFSETOF(JSBoundFunction, m_boundArgsLength); }
    static constexpr ptrdiff_t offsetOfNameMayBeNull() { return OBJECT_OFFSETOF(JSBoundFunction, m_nameMayBeNull); }
    static constexpr ptrdiff_t offsetOfLength() { return OBJECT_OFFSETOF(JSBoundFunction, m_length); }
    static constexpr ptrdiff_t offsetOfCanConstruct() { return OBJECT_OFFSETOF(JSBoundFunction, m_canConstruct); }

    template<typename Functor>
    void forEachBoundArg(const Functor& func)
    {
        unsigned length = boundArgsLength();
        if (!length)
            return;
        if (length <= m_boundArgs.size()) {
            for (unsigned index = 0; index < length; ++index) {
                if (func(m_boundArgs[index].get()) == IterationStatus::Done)
                    return;
            }
            return;
        }
        for (unsigned index = 0; index < length; ++index) {
            if (func(jsCast<JSImmutableButterfly*>(m_boundArgs[0].get())->get(index)) == IterationStatus::Done)
                return;
        }
    }

    bool canConstruct()
    {
        if (m_canConstruct == TriState::Indeterminate)
            return canConstructSlow();
        return m_canConstruct == TriState::True;
    }

    static bool canSkipNameAndLengthMaterialization(JSGlobalObject*, Structure*);

    DECLARE_INFO;

private:
    JSBoundFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, JSObject* targetFunction, JSValue boundThis, unsigned boundArgsLength, JSValue arg0, JSValue arg1, JSValue arg2, JSString* nameMayBeNull, double length);

    JSString* nameSlow(VM&);
    double lengthSlow(VM&);
    bool canConstructSlow();
    String nameStringWithoutGCSlow(VM&);

    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    WriteBarrier<JSObject> m_targetFunction;
    WriteBarrier<Unknown> m_boundThis;
    std::array<WriteBarrier<Unknown>, maxEmbeddedArgs> m_boundArgs { };
    WriteBarrier<JSString> m_nameMayBeNull;
    double m_length { PNaN };
    unsigned m_boundArgsLength { 0 };
    TriState m_canConstruct { TriState::Indeterminate };
};

JSC_DECLARE_HOST_FUNCTION(boundFunctionCall);
JSC_DECLARE_HOST_FUNCTION(boundFunctionConstruct);
JSC_DECLARE_HOST_FUNCTION(boundThisNoArgsFunctionCall);

} // namespace JSC
