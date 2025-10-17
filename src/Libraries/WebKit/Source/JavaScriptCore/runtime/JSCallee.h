/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#include "JSGlobalObject.h"
#include "JSObject.h"
#include "JSScope.h"

namespace JSC {

class JSGlobalObject;
class LLIntOffsetsExtractor;


class JSCallee : public JSNonFinalObject {
    friend class JIT;
#if ENABLE(DFG_JIT)
    friend class DFG::SpeculativeJIT;
    friend class DFG::JITCompiler;
#endif
    friend class VM;

public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | ImplementsHasInstance | ImplementsDefaultHasInstance;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.calleeSpace();
    }

    static JSCallee* create(VM& vm, JSGlobalObject* globalObject, JSScope* scope)
    {
        JSCallee* callee = new (NotNull, allocateCell<JSCallee>(vm)) JSCallee(vm, scope, globalObject->calleeStructure());
        callee->finishCreation(vm);
        return callee;
    }
    
    JSScope* scope()
    {
        return m_scope.get();
    }

    // This method may be called for host functions, in which case it
    // will return an arbitrary value. This should only be used for
    // optimized paths in which the return value does not matter for
    // host functions, and checking whether the function is a host
    // function is deemed too expensive.
    JSScope* scopeUnchecked()
    {
        return m_scope.get();
    }

    void setScope(VM& vm, JSScope* scope)
    {
        m_scope.set(vm, this, scope);
    }

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static constexpr ptrdiff_t offsetOfScopeChain()
    {
        return OBJECT_OFFSETOF(JSCallee, m_scope);
    }

protected:
    JSCallee(VM&, JSGlobalObject*, Structure*);
    JSCallee(VM&, JSScope*, Structure*);

    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

private:
    friend class LLIntOffsetsExtractor;

    WriteBarrier<JSScope> m_scope;
};

} // namespace JSC
