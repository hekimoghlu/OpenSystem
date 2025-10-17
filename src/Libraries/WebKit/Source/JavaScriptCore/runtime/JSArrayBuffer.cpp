/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#include "JSArrayBuffer.h"

#include "JSCInlines.h"
#include "TypedArrayController.h"

namespace JSC {

const ClassInfo JSArrayBuffer::s_info = { "ArrayBuffer"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSArrayBuffer) };

JSArrayBuffer::JSArrayBuffer(VM& vm, Structure* structure, RefPtr<ArrayBuffer>&& arrayBuffer)
    : Base(vm, structure)
    , m_impl(arrayBuffer.get())
{
}

void JSArrayBuffer::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    // This probably causes GCs in the various VMs to overcount the impact of the array buffer.
    vm.heap.addReference(this, impl());
    vm.m_typedArrayController->registerWrapper(globalObject, impl(), this);
}

JSArrayBuffer* JSArrayBuffer::create(
    VM& vm, Structure* structure, RefPtr<ArrayBuffer>&& buffer)
{
    JSArrayBuffer* result =
        new (NotNull, allocateCell<JSArrayBuffer>(vm))
        JSArrayBuffer(vm, structure, WTFMove(buffer));
    result->finishCreation(vm, structure->globalObject());
    return result;
}

Structure* JSArrayBuffer::createStructure(
    VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(
        vm, globalObject, prototype, TypeInfo(ArrayBufferType, StructureFlags), info(),
        NonArray);
}

bool JSArrayBuffer::isShared() const
{
    return impl()->isShared();
}

ArrayBufferSharingMode JSArrayBuffer::sharingMode() const
{
    return impl()->sharingMode();
}

size_t JSArrayBuffer::estimatedSize(JSCell* cell, VM& vm)
{
    JSArrayBuffer* thisObject = jsCast<JSArrayBuffer*>(cell);
    size_t bufferEstimatedSize = thisObject->impl()->gcSizeEstimateInBytes();
    return Base::estimatedSize(cell, vm) + bufferEstimatedSize;
}

} // namespace JSC

