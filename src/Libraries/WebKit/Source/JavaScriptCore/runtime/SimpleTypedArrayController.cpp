/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#include "SimpleTypedArrayController.h"

#include "ArrayBuffer.h"
#include "JSArrayBuffer.h"
#include "JSCJSValueInlines.h"
#include "JSGlobalObject.h"

namespace JSC {

SimpleTypedArrayController::SimpleTypedArrayController(bool allowAtomicsWait)
    : m_allowAtomicsWait(allowAtomicsWait)
{
}

SimpleTypedArrayController::~SimpleTypedArrayController() = default;

JSArrayBuffer* SimpleTypedArrayController::toJS(JSGlobalObject* lexicalGlobalObject, JSGlobalObject* globalObject, ArrayBuffer* native)
{
    UNUSED_PARAM(lexicalGlobalObject);
    if (JSArrayBuffer* buffer = native->m_wrapper.get())
        return buffer;

    // The JSArrayBuffer::create function will register the wrapper in finishCreation.
    JSArrayBuffer* result = JSArrayBuffer::create(globalObject->vm(), globalObject->arrayBufferStructure(native->sharingMode()), native);
    return result;
}

void SimpleTypedArrayController::registerWrapper(JSGlobalObject*, ArrayBuffer* native, JSArrayBuffer* wrapper)
{
    ASSERT(!native->m_wrapper);
    native->m_wrapper = Weak<JSArrayBuffer>(wrapper, &m_owner);
}

bool SimpleTypedArrayController::isAtomicsWaitAllowedOnCurrentThread()
{
    return m_allowAtomicsWait;
}

bool SimpleTypedArrayController::JSArrayBufferOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, JSC::AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    if (UNLIKELY(reason))
        *reason = "JSArrayBuffer is opaque root"_s;
    auto& wrapper = *JSC::jsCast<JSC::JSArrayBuffer*>(handle.slot()->asCell());
    return visitor.containsOpaqueRoot(wrapper.impl());
}

void SimpleTypedArrayController::JSArrayBufferOwner::finalize(JSC::Handle<JSC::Unknown>, void*) { }

} // namespace JSC

