/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#include "WebAssemblyFunctionBase.h"

#if ENABLE(WEBASSEMBLY)

#include "HeapCellInlines.h"
#include "HeapInlines.h"
#include "JSCellInlines.h"
#include "JSWebAssemblyInstance.h"
#include "SlotVisitorInlines.h"
#include "WasmTypeDefinitionInlines.h"

namespace JSC {

const ClassInfo WebAssemblyFunctionBase::s_info = { "WebAssemblyFunctionBase"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyFunctionBase) };

WebAssemblyFunctionBase::WebAssemblyFunctionBase(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure, JSWebAssemblyInstance* instance, Wasm::WasmOrJSImportableFunction&& importableFunction, Wasm::WasmOrJSImportableFunctionCallLinkInfo* callLinkInfo)
    : Base(vm, executable, globalObject, structure)
    , m_importableFunction(WTFMove(importableFunction))
    , m_callLinkInfo(callLinkInfo)
    , m_instance(instance, WriteBarrierEarlyInit)
{ }

template<typename Visitor>
void WebAssemblyFunctionBase::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    WebAssemblyFunctionBase* thisObject = jsCast<WebAssemblyFunctionBase*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_instance);
}

DEFINE_VISIT_CHILDREN(WebAssemblyFunctionBase);

void WebAssemblyFunctionBase::finishCreation(VM& vm, NativeExecutable* executable, unsigned length, const String& name)
{
    Base::finishCreation(vm, executable, length, name);
    ASSERT(inherits(info()));
}

const Wasm::FunctionSignature& WebAssemblyFunctionBase::signature() const
{
    return Wasm::TypeInformation::getFunctionSignature(typeIndex());
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
