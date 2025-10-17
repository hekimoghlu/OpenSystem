/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/InternalFunction.h>

namespace WebCore {

JSC_DECLARE_HOST_FUNCTION(callThrowTypeErrorForJSDOMConstructor);
JSC_DECLARE_HOST_FUNCTION(callThrowTypeErrorForJSDOMConstructorNotConstructable);

// Base class for all callable constructor objects in the JSC bindings.
class JSDOMConstructorBase : public JSC::InternalFunction {
public:
    using Base = InternalFunction;

    static constexpr unsigned StructureFlags = Base::StructureFlags;
    static constexpr JSC::DestructionMode needsDestruction = JSC::DoesNotNeedDestruction;

    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(JSDOMConstructorBase));
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(CellType, JSDOMConstructorBase);
        static_assert(CellType::destroy == JSC::JSCell::destroy, "JSDOMConstructor<JSClass> is not destructible actually");
        return subspaceForImpl(vm);
    }

    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);

    JSDOMGlobalObject* globalObject() const { return JSC::jsCast<JSDOMGlobalObject*>(Base::globalObject()); }
    ScriptExecutionContext* scriptExecutionContext() const { return globalObject()->scriptExecutionContext(); }

protected:
    JSDOMConstructorBase(JSC::VM& vm, JSC::Structure* structure, JSC::NativeFunction functionForConstruct)
        : Base(vm, structure,
            functionForConstruct ? callThrowTypeErrorForJSDOMConstructor : callThrowTypeErrorForJSDOMConstructorNotConstructable,
            functionForConstruct ? functionForConstruct : callThrowTypeErrorForJSDOMConstructorNotConstructable)
    {
    }
};

} // namespace WebCore
