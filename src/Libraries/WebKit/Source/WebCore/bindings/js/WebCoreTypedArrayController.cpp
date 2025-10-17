/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#include "WebCoreTypedArrayController.h"

#include "JSDOMConvertBufferSource.h"
#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/JSArrayBuffer.h>
#include <JavaScriptCore/JSCInlines.h>

namespace WebCore {

WebCoreTypedArrayController::WebCoreTypedArrayController(bool allowAtomicsWait)
    : m_allowAtomicsWait(allowAtomicsWait)
{
}

WebCoreTypedArrayController::~WebCoreTypedArrayController() = default;

JSC::JSArrayBuffer* WebCoreTypedArrayController::toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSC::JSGlobalObject* globalObject, JSC::ArrayBuffer* buffer)
{
    return JSC::jsCast<JSC::JSArrayBuffer*>(WebCore::toJS(lexicalGlobalObject, JSC::jsCast<JSDOMGlobalObject*>(globalObject), buffer));
}

void WebCoreTypedArrayController::registerWrapper(JSC::JSGlobalObject* globalObject, JSC::ArrayBuffer* native, JSC::JSArrayBuffer* wrapper)
{
    cacheWrapper(JSC::jsCast<JSDOMGlobalObject*>(globalObject)->world(), native, wrapper);
}

bool WebCoreTypedArrayController::isAtomicsWaitAllowedOnCurrentThread()
{
    return m_allowAtomicsWait;
}

bool WebCoreTypedArrayController::JSArrayBufferOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, JSC::AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    if (UNLIKELY(reason))
        *reason = "ArrayBuffer is opaque root"_s;
    auto& wrapper = *JSC::jsCast<JSC::JSArrayBuffer*>(handle.slot()->asCell());
    return visitor.containsOpaqueRoot(wrapper.impl());
}

void WebCoreTypedArrayController::JSArrayBufferOwner::finalize(JSC::Handle<JSC::Unknown> handle, void* context)
{
    auto& wrapper = *static_cast<JSC::JSArrayBuffer*>(handle.slot()->asCell());
    uncacheWrapper(*static_cast<DOMWrapperWorld*>(context), wrapper.impl(), &wrapper);
}

} // namespace WebCore
