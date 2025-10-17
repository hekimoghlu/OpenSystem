/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#include "Exception.h"

#include "Interpreter.h"
#include "JSCJSValueInlines.h"
#include "JSObjectInlines.h"
#include "StructureInlines.h"
#include "JSWebAssemblyException.h"
#include "WasmTag.h"

namespace JSC {

const ClassInfo Exception::s_info = { "Exception"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(Exception) };

Exception* Exception::create(VM& vm, JSValue thrownValue, StackCaptureAction action)
{
    Exception* result = new (NotNull, allocateCell<Exception>(vm)) Exception(vm, thrownValue);
    result->finishCreation(vm, action);
    return result;
}

void Exception::destroy(JSCell* cell)
{
    Exception* exception = static_cast<Exception*>(cell);
    exception->~Exception();
}

Structure* Exception::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(CellType, StructureFlags), info());
}

template<typename Visitor>
void Exception::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    Exception* thisObject = jsCast<Exception*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.append(thisObject->m_value);
    for (StackFrame& frame : thisObject->m_stack)
        frame.visitAggregate(visitor);
    visitor.reportExtraMemoryVisited(thisObject->m_stack.sizeInBytes());
}

DEFINE_VISIT_CHILDREN(Exception);

Exception::Exception(VM& vm, JSValue thrownValue)
    : Base(vm, vm.exceptionStructure.get())
    , m_value(thrownValue, WriteBarrierEarlyInit)
{
}

Exception::~Exception() = default;

void Exception::finishCreation(VM& vm, StackCaptureAction action)
{
    Base::finishCreation(vm);

    Vector<StackFrame> stackTrace;
    if (action == StackCaptureAction::CaptureStack)
        vm.interpreter.getStackTrace(this, stackTrace, 0, Options::exceptionStackTraceLimit());
    m_stack = WTFMove(stackTrace);
    vm.heap.reportExtraMemoryAllocated(this, m_stack.sizeInBytes());
}

#if ENABLE(WEBASSEMBLY)

void Exception::wrapValueForJSTag(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    FixedVector<uint64_t> payload { static_cast<uint64_t>(JSValue::encode(m_value.get())) };
    auto* wrapper = JSWebAssemblyException::create(vm, globalObject->webAssemblyExceptionStructure(), Wasm::Tag::jsExceptionTag(), WTFMove(payload));
    m_value.set(vm, this, wrapper);
}

#endif

} // namespace JSC
