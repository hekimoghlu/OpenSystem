/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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
#include "JSErrorHandler.h"

#include "Document.h"
#include "ErrorEvent.h"
#include "Event.h"
#include "JSDOMConvertNumbers.h"
#include "JSDOMConvertStrings.h"
#include "JSDOMWindow.h"
#include "JSEvent.h"
#include "JSExecState.h"
#include "JSExecStateInstrumentation.h"
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/VMEntryScopeInlines.h>
#include <wtf/Ref.h>

namespace WebCore {
using namespace JSC;

inline JSErrorHandler::JSErrorHandler(JSObject& listener, JSObject& wrapper, bool isAttribute, DOMWrapperWorld& world)
    : JSEventListener(&listener, &wrapper, isAttribute, CreatedFromMarkup::No, world)
{
}

Ref<JSErrorHandler> JSErrorHandler::create(JSC::JSObject& listener, JSC::JSObject& wrapper, bool isAttribute, DOMWrapperWorld& world)
{
    return adoptRef(*new JSErrorHandler(listener, wrapper, isAttribute, world));
}

JSErrorHandler::~JSErrorHandler() = default;

void JSErrorHandler::handleEvent(ScriptExecutionContext& scriptExecutionContext, Event& event)
{
    auto* errorEvent = dynamicDowncast<ErrorEvent>(event);
    if (!errorEvent)
        return JSEventListener::handleEvent(scriptExecutionContext, event);

    VM& vm = scriptExecutionContext.vm();
    JSLockHolder lock(vm);

    JSObject* jsFunction = this->ensureJSFunction(scriptExecutionContext);
    if (!jsFunction)
        return;

    auto* isolatedWorld = this->isolatedWorld();
    if (UNLIKELY(!isolatedWorld))
        return;

    auto* globalObject = toJSDOMGlobalObject(scriptExecutionContext, *isolatedWorld);
    if (!globalObject)
        return;

    auto callData = JSC::getCallData(jsFunction);
    if (callData.type != CallData::Type::None) {
        Ref<JSErrorHandler> protectedThis(*this);

        RefPtr<Event> savedEvent;
        auto* jsFunctionWindow = jsDynamicCast<JSDOMWindow*>(jsFunction->globalObject());
        if (jsFunctionWindow) {
            savedEvent = jsFunctionWindow->currentEvent();

            // window.event should not be set when the target is inside a shadow tree, as per the DOM specification.
            if (!errorEvent->currentTargetIsInShadowTree())
                jsFunctionWindow->setCurrentEvent(errorEvent);
        }

        MarkedArgumentBuffer args;
        args.append(toJS<IDLDOMString>(*globalObject, errorEvent->message()));
        args.append(toJS<IDLUSVString>(*globalObject, errorEvent->filename()));
        args.append(toJS<IDLUnsignedLong>(errorEvent->lineno()));
        args.append(toJS<IDLUnsignedLong>(errorEvent->colno()));
        args.append(errorEvent->error(*globalObject));
        ASSERT(!args.hasOverflowed());

        VM& vm = globalObject->vm();
        VMEntryScope entryScope(vm, vm.entryScope ? vm.entryScope->globalObject() : globalObject);

        JSExecState::instrumentFunction(&scriptExecutionContext, callData);

        NakedPtr<JSC::Exception> exception;
        JSValue returnValue = JSExecState::profiledCall(globalObject, JSC::ProfilingReason::Other, jsFunction, callData, globalObject, args, exception);

        InspectorInstrumentation::didCallFunction(&scriptExecutionContext);

        if (exception)
            reportException(jsFunction->globalObject(), exception);
        else {
            if (returnValue.isTrue())
                errorEvent->preventDefault();
        }

        if (jsFunctionWindow)
            jsFunctionWindow->setCurrentEvent(savedEvent.get());
    }
}

} // namespace WebCore
