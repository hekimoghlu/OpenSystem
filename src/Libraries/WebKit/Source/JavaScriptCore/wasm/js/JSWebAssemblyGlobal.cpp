/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
#include "config.h"
#include "JSWebAssemblyGlobal.h"
#include "ObjectConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSWebAssemblyGlobal::s_info = { "WebAssembly.Global"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWebAssemblyGlobal) };

JSWebAssemblyGlobal* JSWebAssemblyGlobal::create(VM& vm, Structure* structure, Ref<Wasm::Global>&& global)
{
    auto* instance = new (NotNull, allocateCell<JSWebAssemblyGlobal>(vm)) JSWebAssemblyGlobal(vm, structure, WTFMove(global));
    instance->global()->setOwner(instance);
    instance->finishCreation(vm);
    return instance;
}

Structure* JSWebAssemblyGlobal::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

JSWebAssemblyGlobal::JSWebAssemblyGlobal(VM& vm, Structure* structure, Ref<Wasm::Global>&& global)
    : Base(vm, structure)
    , m_global(WTFMove(global))
{
}

void JSWebAssemblyGlobal::destroy(JSCell* cell)
{
    static_cast<JSWebAssemblyGlobal*>(cell)->JSWebAssemblyGlobal::~JSWebAssemblyGlobal();
}

template<typename Visitor>
void JSWebAssemblyGlobal::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSWebAssemblyGlobal* thisObject = jsCast<JSWebAssemblyGlobal*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());

    Base::visitChildren(thisObject, visitor);
    thisObject->global()->visitAggregate(visitor);
}

JSObject* JSWebAssemblyGlobal::type(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();

    JSObject* result = constructEmptyObject(globalObject, globalObject->objectPrototype(), 2);

    result->putDirect(vm, Identifier::fromString(vm, "mutable"_s), jsBoolean(m_global->mutability() == Wasm::Mutable));

    Wasm::Type valueType = m_global->type();
    JSString* valueString = typeToJSAPIString(vm, valueType);
    if (!valueString)
        return nullptr;
    result->putDirect(vm, Identifier::fromString(vm, "value"_s), valueString);

    return result;
}

DEFINE_VISIT_CHILDREN(JSWebAssemblyGlobal);

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
