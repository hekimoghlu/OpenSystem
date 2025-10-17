/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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

template<typename JSClass> class JSDOMConstructor final : public JSDOMConstructorBase {
public:
    using Base = JSDOMConstructorBase;

    static JSDOMConstructor* create(JSC::VM&, JSC::Structure*, JSDOMGlobalObject&);
    static JSC::Structure* createStructure(JSC::VM&, JSC::JSGlobalObject&, JSC::JSValue prototype);

    DECLARE_INFO;

    // Must be defined for each specialization class.
    static JSC::JSValue prototypeForStructure(JSC::VM&, const JSDOMGlobalObject&);

    // Must be defined for each specialization class.
    static JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES construct(JSC::JSGlobalObject*, JSC::CallFrame*);

private:
    JSDOMConstructor(JSC::VM& vm, JSC::Structure* structure)
        : Base(vm, structure, construct)
    {
    }

    void finishCreation(JSC::VM&, JSDOMGlobalObject&);

    // Usually defined for each specialization class.
    void initializeProperties(JSC::VM&, JSDOMGlobalObject&) { }
};

template<typename JSClass> inline JSDOMConstructor<JSClass>* JSDOMConstructor<JSClass>::create(JSC::VM& vm, JSC::Structure* structure, JSDOMGlobalObject& globalObject)
{
    JSDOMConstructor* constructor = new (NotNull, JSC::allocateCell<JSDOMConstructor>(vm)) JSDOMConstructor(vm, structure);
    constructor->finishCreation(vm, globalObject);
    return constructor;
}

template<typename JSClass> inline JSC::Structure* JSDOMConstructor<JSClass>::createStructure(JSC::VM& vm, JSC::JSGlobalObject& globalObject, JSC::JSValue prototype)
{
    auto* structure = JSC::Structure::create(vm, &globalObject, prototype, JSC::TypeInfo(JSC::InternalFunctionType, StructureFlags), info());
    structure->setMayBePrototype(true);
    return structure;
}

template<typename JSClass> inline void JSDOMConstructor<JSClass>::finishCreation(JSC::VM& vm, JSDOMGlobalObject& globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    initializeProperties(vm, globalObject);
}

} // namespace WebCore
