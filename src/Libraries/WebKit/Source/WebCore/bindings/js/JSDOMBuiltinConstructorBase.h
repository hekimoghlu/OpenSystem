/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

#include "JSDOMConstructorBase.h"

namespace WebCore {

class JSDOMBuiltinConstructorBase : public JSDOMConstructorBase {
public:
    using Base = JSDOMConstructorBase;

    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(JSDOMBuiltinConstructorBase));
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(CellType, JSDOMBuiltinConstructorBase);
        static_assert(CellType::destroy == JSC::JSCell::destroy, "JSDOMBuiltinConstructor<JSClass> is not destructible actually");
        return subspaceForImpl(vm);
    }

protected:
    JSDOMBuiltinConstructorBase(JSC::VM& vm, JSC::Structure* structure, JSC::NativeFunction functionForConstruct)
        : Base(vm, structure, functionForConstruct)
    {
    }

    DECLARE_VISIT_CHILDREN;

    JSC::JSFunction* initializeFunction();
    void setInitializeFunction(JSC::VM&, JSC::JSFunction&);

private:
    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);

    JSC::WriteBarrier<JSC::JSFunction> m_initializeFunction;
};

inline JSC::JSFunction* JSDOMBuiltinConstructorBase::initializeFunction()
{
    return m_initializeFunction.get();
}

inline void JSDOMBuiltinConstructorBase::setInitializeFunction(JSC::VM& vm, JSC::JSFunction& function)
{
    m_initializeFunction.set(vm, this, &function);
}


} // namespace WebCore
