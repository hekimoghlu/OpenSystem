/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "JSWorkerGlobalScope.h"

#include "JSDOMExceptionHandling.h"
#include "WebCoreOpaqueRootInlines.h"
#include "WorkerGlobalScope.h"
#include "WorkerLocation.h"
#include "WorkerNavigator.h"
#include <JavaScriptCore/Error.h>

namespace WebCore {
using namespace JSC;

template<typename Visitor>
void JSWorkerGlobalScope::visitAdditionalChildren(Visitor& visitor)
{
    if (auto* location = wrapped().optionalLocation())
        addWebCoreOpaqueRoot(visitor, *location);
    if (auto* navigator = wrapped().optionalNavigator())
        addWebCoreOpaqueRoot(visitor, *navigator);
    ScriptExecutionContext& context = wrapped();
    addWebCoreOpaqueRoot(visitor, context);
    
    // Normally JSEventTargetCustom.cpp's JSEventTarget::visitAdditionalChildren() would call this. But
    // even though WorkerGlobalScope is an EventTarget, JSWorkerGlobalScope does not subclass
    // JSEventTarget, so we need to do this here.
    wrapped().visitJSEventListeners(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSWorkerGlobalScope);

JSValue JSWorkerGlobalScope::queueMicrotask(JSGlobalObject& lexicalGlobalObject, CallFrame& callFrame)
{
    VM& vm = lexicalGlobalObject.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(callFrame.argumentCount() < 1))
        return throwException(&lexicalGlobalObject, scope, createNotEnoughArgumentsError(&lexicalGlobalObject));

    JSValue functionValue = callFrame.uncheckedArgument(0);
    if (UNLIKELY(!functionValue.isCallable()))
        return JSValue::decode(throwArgumentMustBeFunctionError(lexicalGlobalObject, scope, 0, "callback"_s, "WorkerGlobalScope"_s, "queueMicrotask"_s));

    scope.release();
    globalObjectMethodTable()->queueMicrotaskToEventLoop(*this, JSC::QueuedTask { nullptr, this, functionValue, { }, { }, { }, { } });
    return jsUndefined();
}

} // namespace WebCore
