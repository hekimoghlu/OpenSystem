/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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

#include "JSDOMConstructorWithDocument.h"

namespace WebCore {

// All legacy factory functions are exposed on Window only, while authors are advised against adding new ones.
template<typename JSClass> class JSDOMLegacyFactoryFunction final : public JSDOMConstructorWithDocument {
public:
    using Base = JSDOMConstructorWithDocument;

    static JSDOMLegacyFactoryFunction* create(JSC::VM&, JSC::Structure*, JSDOMGlobalObject&);
    static JSC::Structure* createStructure(JSC::VM&, JSC::JSGlobalObject&, JSC::JSValue prototype);

    DECLARE_INFO;

    // Must be defined for each specialization class.
    static JSC::JSValue prototypeForStructure(JSC::VM&, const JSDOMGlobalObject&);

    // Must be defined for each specialization class.
    static JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES construct(JSC::JSGlobalObject*, JSC::CallFrame*);

private:
    JSDOMLegacyFactoryFunction(JSC::VM& vm, JSC::Structure* structure)
        : Base(vm, structure, construct)
    { 
    }

    void finishCreation(JSC::VM&, JSDOMGlobalObject&);

    // Usually defined for each specialization class.
    void initializeProperties(JSC::VM&, JSDOMGlobalObject&) { }
};

template<typename JSClass> inline JSDOMLegacyFactoryFunction<JSClass>* JSDOMLegacyFactoryFunction<JSClass>::create(JSC::VM& vm, JSC::Structure* structure, JSDOMGlobalObject& globalObject)
{
    JSDOMLegacyFactoryFunction* constructor = new (NotNull, JSC::allocateCell<JSDOMLegacyFactoryFunction>(vm)) JSDOMLegacyFactoryFunction(vm, structure);
    constructor->finishCreation(vm, globalObject);
    return constructor;
}

template<typename JSClass> inline JSC::Structure* JSDOMLegacyFactoryFunction<JSClass>::createStructure(JSC::VM& vm, JSC::JSGlobalObject& globalObject, JSC::JSValue prototype)
{
    return JSC::Structure::create(vm, &globalObject, prototype, JSC::TypeInfo(JSC::InternalFunctionType, StructureFlags), info());
}

template<typename JSClass> inline void JSDOMLegacyFactoryFunction<JSClass>::finishCreation(JSC::VM& vm, JSDOMGlobalObject& globalObject)
{
    Base::finishCreation(globalObject);
    ASSERT(inherits(info()));
    initializeProperties(vm, globalObject);
}

} // namespace WebCore
