/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "DOMConstructors.h"
#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/JSObjectInlines.h>

namespace WebCore {

inline JSC::Structure* JSDOMGlobalObject::createStructure(JSC::VM& vm, JSC::JSValue prototype)
{
    return JSC::Structure::create(vm, 0, prototype, JSC::TypeInfo(JSC::GlobalObjectType, StructureFlags), info());
}

inline JSDOMStructureMap& JSDOMGlobalObject::structures(NoLockingNecessaryTag)
{
    ASSERT(!vm().heap.mutatorShouldBeFenced());
    IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
    return m_structures;
    IGNORE_CLANG_WARNINGS_END
}

inline DOMGuardedObjectSet& JSDOMGlobalObject::guardedObjects(NoLockingNecessaryTag)
{
    ASSERT(!vm().heap.mutatorShouldBeFenced());
    IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
    return m_guardedObjects;
    IGNORE_CLANG_WARNINGS_END
}

template<class ConstructorClass, DOMConstructorID constructorID>
inline JSC::JSObject* getDOMConstructor(JSC::VM& vm, const JSDOMGlobalObject& globalObject)
{
    // No locking is necessary unless we need to add a new constructor to JSDOMGlobalObject::constructors().
    if (JSC::JSObject* constructor = globalObject.constructors().array()[static_cast<unsigned>(constructorID)].get())
        return constructor;
    JSC::JSObject* constructor = ConstructorClass::create(vm, ConstructorClass::createStructure(vm, const_cast<JSDOMGlobalObject&>(globalObject), ConstructorClass::prototypeForStructure(vm, globalObject)), const_cast<JSDOMGlobalObject&>(globalObject));
    ASSERT(!globalObject.constructors().array()[static_cast<unsigned>(constructorID)].get());
    JSDOMGlobalObject& mutableGlobalObject = const_cast<JSDOMGlobalObject&>(globalObject);
    mutableGlobalObject.constructors().array()[static_cast<unsigned>(constructorID)].set(vm, &globalObject, constructor);
    return constructor;
}

template<class JSClass>
JSClass* toJSDOMGlobalObject(JSC::VM&, JSC::JSValue value)
{
    static_assert(std::is_base_of_v<JSDOMGlobalObject, JSClass>);

    if (auto* object = value.getObject()) {
        if (object->type() == JSC::GlobalProxyType)
            return JSC::jsDynamicCast<JSClass*>(JSC::jsCast<JSC::JSGlobalProxy*>(object)->target());
        if (object->inherits<JSClass>())
            return JSC::jsCast<JSClass*>(object);
    }

    return nullptr;
}


} // namespace WebCore
