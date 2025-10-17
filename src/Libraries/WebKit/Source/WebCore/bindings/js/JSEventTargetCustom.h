/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#pragma once

#include "JSDOMBinding.h"
#include "JSDOMBindingSecurity.h"
#include "JSDOMOperation.h"
#include "LocalDOMWindow.h"

namespace WebCore {

// Wrapper type for JSEventTarget's castedThis because JSDOMWindow and JSWorkerGlobalScope do not inherit JSEventTarget.
class JSEventTargetWrapper {
public:
    JSEventTargetWrapper() = default;

    JSEventTargetWrapper(EventTarget& wrapped, JSC::JSObject& wrapper)
        : m_wrapped(&wrapped)
        , m_wrapper(&wrapper)
    { }

    bool isNull() const { return !m_wrapped; }
    EventTarget& wrapped() { ASSERT(m_wrapped); return *m_wrapped; }

    operator JSC::JSObject&() { ASSERT(m_wrapper); return *m_wrapper; }

private:
    EventTarget* m_wrapped { nullptr };
    JSC::JSObject* m_wrapper { nullptr };
};

JSEventTargetWrapper jsEventTargetCast(JSC::VM&, JSC::JSValue thisValue);

template<> class IDLOperation<JSEventTarget> {
public:
    using ClassParameter = JSEventTargetWrapper*;
    using Operation = JSC::EncodedJSValue(JSC::JSGlobalObject*, JSC::CallFrame*, ClassParameter);

    template<Operation operation, CastedThisErrorBehavior = CastedThisErrorBehavior::Throw>
    static JSC::EncodedJSValue call(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char* operationName)
    {
        auto& vm = JSC::getVM(&lexicalGlobalObject);
        auto throwScope = DECLARE_THROW_SCOPE(vm);

        auto thisValue = callFrame.thisValue().toThis(&lexicalGlobalObject, JSC::ECMAMode::strict());
        auto thisObject = jsEventTargetCast(vm, thisValue.isUndefinedOrNull() ? JSC::JSValue(&lexicalGlobalObject) : thisValue);
        if (UNLIKELY(thisObject.isNull()))
            return throwThisTypeError(lexicalGlobalObject, throwScope, "EventTarget", operationName);

        auto& wrapped = thisObject.wrapped();
        if (auto window = dynamicDowncast<LocalDOMWindow>(wrapped)) {
            if (!window->frame() || !BindingSecurity::shouldAllowAccessToDOMWindow(&lexicalGlobalObject, *window, ThrowSecurityError))
                return JSC::JSValue::encode(JSC::jsUndefined());
        }

        RELEASE_AND_RETURN(throwScope, (operation(&lexicalGlobalObject, &callFrame, &thisObject)));
    }

};


} // namespace WebCore
