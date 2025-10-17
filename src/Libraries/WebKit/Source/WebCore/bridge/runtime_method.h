/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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

#include "BridgeJSC.h"
#include <JavaScriptCore/InternalFunction.h>
#include <JavaScriptCore/JSGlobalObject.h>

namespace JSC {

class WEBCORE_EXPORT RuntimeMethod : public InternalFunction {
public:
    using Base = InternalFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetCallData;

    template<typename CellType, JSC::SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(RuntimeMethod), "RuntimeMethod subclasses that add fields need to override subspaceFor<>()");
        static_assert(CellType::destroy == JSC::JSCell::destroy);
        return subspaceForImpl(vm);
    }
    
    static RuntimeMethod* create(JSGlobalObject*, JSGlobalObject* globalObject, Structure* structure, const String& name, Bindings::Method* method)
    {
        VM& vm = globalObject->vm();
        RuntimeMethod* runtimeMethod = new (NotNull, allocateCell<RuntimeMethod>(vm)) RuntimeMethod(vm, structure, method);
        runtimeMethod->finishCreation(vm, name);
        return runtimeMethod;
    }

    Bindings::Method* method() const { return m_method; }

    DECLARE_INFO;

    static FunctionPrototype* createPrototype(VM&, JSGlobalObject& globalObject)
    {
        return globalObject.functionPrototype();
    }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
    }

protected:
    RuntimeMethod(VM&, Structure*, Bindings::Method*);
    void finishCreation(VM&, const String&);

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);

private:
    static GCClient::IsoSubspace* subspaceForImpl(VM&);

    Bindings::Method* m_method;
};

} // namespace JSC
