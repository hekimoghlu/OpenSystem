/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#include "JSEventTarget.h"

#include "EventTarget.h"
#include "EventTargetHeaders.h"
#include "EventTargetInterfaces.h"
#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMWindow.h"
#include "JSEventListener.h"
#include "JSWindowProxy.h"
#include "JSWorkerGlobalScope.h"
#include "LocalDOMWindow.h"
#include "WorkerGlobalScope.h"

#if ENABLE(OFFSCREEN_CANVAS)
#include "OffscreenCanvas.h"
#endif

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<EventTarget>&& value)
{
    return createWrapper<EventTarget>(globalObject, WTFMove(value));
}

EventTarget* JSEventTarget::toWrapped(VM&, JSValue value)
{
    if (value.inherits<JSWindowProxy>())
        return &jsCast<JSWindowProxy*>(asObject(value))->wrapped();
    if (value.inherits<JSDOMWindow>())
        return &jsCast<JSDOMWindow*>(asObject(value))->wrapped();
    if (value.inherits<JSWorkerGlobalScope>())
        return &jsCast<JSWorkerGlobalScope*>(asObject(value))->wrapped();
    if (value.inherits<JSEventTarget>())
        return &jsCast<JSEventTarget*>(asObject(value))->wrapped();
    return nullptr;
}

JSEventTargetWrapper jsEventTargetCast(VM& vm, JSValue thisValue)
{
    if (auto* target = jsDynamicCast<JSEventTarget*>(thisValue))
        return { target->wrapped(), *target };
    if (auto* window = toJSDOMGlobalObject<JSDOMWindow>(vm, thisValue))
        return { window->wrapped(), *window };
    if (auto* scope = toJSDOMGlobalObject<JSWorkerGlobalScope>(vm, thisValue))
        return { scope->wrapped(), *scope };
    return { };
}

template<typename Visitor>
void JSEventTarget::visitAdditionalChildren(Visitor& visitor)
{
    wrapped().visitJSEventListeners(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSEventTarget);

} // namespace WebCore
