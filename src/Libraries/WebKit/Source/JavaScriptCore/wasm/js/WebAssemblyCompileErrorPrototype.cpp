/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#include "WebAssemblyCompileErrorPrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo WebAssemblyCompileErrorPrototype::s_info = { "WebAssembly.CompileError"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyCompileErrorPrototype) };

WebAssemblyCompileErrorPrototype* WebAssemblyCompileErrorPrototype::create(VM& vm, JSGlobalObject*, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyCompileErrorPrototype>(vm)) WebAssemblyCompileErrorPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* WebAssemblyCompileErrorPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyCompileErrorPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    putDirectWithoutTransition(vm, vm.propertyNames->name, jsNontrivialString(vm, "CompileError"_s), static_cast<unsigned>(PropertyAttribute::DontEnum));
    putDirectWithoutTransition(vm, vm.propertyNames->message, jsEmptyString(vm), static_cast<unsigned>(PropertyAttribute::DontEnum));
}

WebAssemblyCompileErrorPrototype::WebAssemblyCompileErrorPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
