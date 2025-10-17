/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "JSNativeStdFunction.h"

#include "JSCJSValueInlines.h"
#include "VM.h"

namespace JSC {

const ClassInfo JSNativeStdFunction::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSNativeStdFunction) };

static JSC_DECLARE_HOST_FUNCTION(runStdFunction);

JSNativeStdFunction::JSNativeStdFunction(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure, NativeStdFunction&& function)
    : Base(vm, executable, globalObject, structure)
    , m_function(WTFMove(function))
{
}

template<typename Visitor>
void JSNativeStdFunction::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSNativeStdFunction* thisObject = jsCast<JSNativeStdFunction*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSNativeStdFunction);

void JSNativeStdFunction::finishCreation(VM& vm, NativeExecutable* executable, unsigned length, const String& name)
{
    Base::finishCreation(vm, executable, length, name);
    ASSERT(inherits(info()));
}

JSC_DEFINE_HOST_FUNCTION(runStdFunction, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    JSNativeStdFunction* function = jsCast<JSNativeStdFunction*>(callFrame->jsCallee());
    ASSERT(function);
    return function->function()(globalObject, callFrame);
}

JSNativeStdFunction* JSNativeStdFunction::create(VM& vm, JSGlobalObject* globalObject, unsigned length, const String& name, NativeStdFunction&& nativeStdFunction, Intrinsic intrinsic, NativeFunction nativeConstructor)
{
    NativeExecutable* executable = vm.getHostFunction(runStdFunction, ImplementationVisibility::Private, intrinsic, nativeConstructor, nullptr, name);
    Structure* structure = globalObject->nativeStdFunctionStructure();
    JSNativeStdFunction* function = new (NotNull, allocateCell<JSNativeStdFunction>(vm)) JSNativeStdFunction(vm, executable, globalObject, structure, WTFMove(nativeStdFunction));
    function->finishCreation(vm, executable, length, name);
    return function;
}

} // namespace JSC
