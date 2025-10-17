/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#include "JSDOMWrapper.h"
#include "WebCoreJSClientData.h"

namespace WebCore {

template<typename JSClass> class JSDOMConstructorNotCallable : public JSDOMObject {
public:
    using Base = JSDOMObject;

    static constexpr unsigned StructureFlags = Base::StructureFlags;
    static constexpr JSC::DestructionMode needsDestruction = JSC::DoesNotNeedDestruction;
    static JSDOMConstructorNotCallable* create(JSC::VM&, JSC::Structure*, JSDOMGlobalObject&);
    static JSC::Structure* createStructure(JSC::VM&, JSC::JSGlobalObject&, JSC::JSValue prototype);

    DECLARE_INFO;

    static JSC::JSValue prototypeForStructure(JSC::VM&, const JSDOMGlobalObject&);

    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        static_assert(sizeof(CellType) == sizeof(JSDOMConstructorNotCallable));
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(CellType, JSDOMConstructorNotCallable);
        static_assert(CellType::destroy == JSC::JSCell::destroy, "JSDOMConstructorNotCallable<JSClass> is not destructible actually");
        return &static_cast<JSVMClientData*>(vm.clientData)->domNamespaceObjectSpace();
    }

private:
    JSDOMConstructorNotCallable(JSC::Structure* structure, JSDOMGlobalObject& globalObject)
        : Base(structure, globalObject)
    {
    }

    void finishCreation(JSC::VM&, JSDOMGlobalObject&);
    void initializeProperties(JSC::VM&, JSDOMGlobalObject&) { }
};

template<typename JSClass> inline JSDOMConstructorNotCallable<JSClass>* JSDOMConstructorNotCallable<JSClass>::create(JSC::VM& vm, JSC::Structure* structure, JSDOMGlobalObject& globalObject)
{
    JSDOMConstructorNotCallable* constructor = new (NotNull, JSC::allocateCell<JSDOMConstructorNotCallable>(vm)) JSDOMConstructorNotCallable(structure, globalObject);
    constructor->finishCreation(vm, globalObject);
    return constructor;
}

template<typename JSClass> inline JSC::Structure* JSDOMConstructorNotCallable<JSClass>::createStructure(JSC::VM& vm, JSC::JSGlobalObject& globalObject, JSC::JSValue prototype)
{
    auto* structure = JSC::Structure::create(vm, &globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    structure->setMayBePrototype(true);
    return structure;
}

template<typename JSClass> inline void JSDOMConstructorNotCallable<JSClass>::finishCreation(JSC::VM& vm, JSDOMGlobalObject& globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    initializeProperties(vm, globalObject);
}

} // namespace WebCore
