/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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

#include "IDLTypes.h"
#include "JSCallbackData.h"
#include "TestCallbackFunction.h"
#include <wtf/Forward.h>

namespace WebCore {

class JSTestCallbackFunction final : public TestCallbackFunction {
public:
    static Ref<JSTestCallbackFunction> create(JSC::JSObject* callback, JSDOMGlobalObject* globalObject)
    {
        return adoptRef(*new JSTestCallbackFunction(callback, globalObject));
    }

    ScriptExecutionContext* scriptExecutionContext() const { return ContextDestructionObserver::scriptExecutionContext(); }

    ~JSTestCallbackFunction() final;
    JSCallbackData* callbackData() { return m_data; }

    // Functions
    CallbackResult<typename IDLDOMString::CallbackReturnType> handleEvent(typename IDLLong::ParameterType argument) override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> handleEventRethrowingException(typename IDLLong::ParameterType argument) override;

private:
    JSTestCallbackFunction(JSC::JSObject*, JSDOMGlobalObject*);

    bool hasCallback() const final { return m_data && m_data->callback(); }

    void visitJSFunction(JSC::AbstractSlotVisitor&) override;

    void visitJSFunction(JSC::SlotVisitor&) override;

    JSCallbackData* m_data;
};

JSC::JSValue toJS(TestCallbackFunction&);
inline JSC::JSValue toJS(TestCallbackFunction* impl) { return impl ? toJS(*impl) : JSC::jsNull(); }

template<> struct JSDOMCallbackConverterTraits<JSTestCallbackFunction> {
    using Base = TestCallbackFunction;
};
} // namespace WebCore
