/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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

#if ENABLE(TEST_CONDITIONAL)

#include "IDLTypes.h"
#include "JSCallbackData.h"
#include "TestVoidCallbackFunction.h"
#include <wtf/Forward.h>

namespace WebCore {

class JSTestVoidCallbackFunction final : public TestVoidCallbackFunction {
public:
    static Ref<JSTestVoidCallbackFunction> create(JSC::JSObject* callback, JSDOMGlobalObject* globalObject)
    {
        return adoptRef(*new JSTestVoidCallbackFunction(callback, globalObject));
    }

    ScriptExecutionContext* scriptExecutionContext() const { return ContextDestructionObserver::scriptExecutionContext(); }

    ~JSTestVoidCallbackFunction() final;
    JSCallbackData* callbackData() { return m_data; }

    // Functions
    CallbackResult<typename IDLUndefined::CallbackReturnType> handleEvent(typename IDLFloat32Array::ParameterType arrayParam, typename IDLSerializedScriptValue<SerializedScriptValue>::ParameterType srzParam, typename IDLDOMString::ParameterType strArg, typename IDLBoolean::ParameterType boolParam, typename IDLLong::ParameterType longParam, typename IDLInterface<TestNode>::ParameterType testNodeParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> handleEventRethrowingException(typename IDLFloat32Array::ParameterType arrayParam, typename IDLSerializedScriptValue<SerializedScriptValue>::ParameterType srzParam, typename IDLDOMString::ParameterType strArg, typename IDLBoolean::ParameterType boolParam, typename IDLLong::ParameterType longParam, typename IDLInterface<TestNode>::ParameterType testNodeParam) override;

private:
    JSTestVoidCallbackFunction(JSC::JSObject*, JSDOMGlobalObject*);

    bool hasCallback() const final { return m_data && m_data->callback(); }

    void visitJSFunction(JSC::AbstractSlotVisitor&) override;

    void visitJSFunction(JSC::SlotVisitor&) override;

    JSCallbackData* m_data;
};

JSC::JSValue toJS(TestVoidCallbackFunction&);
inline JSC::JSValue toJS(TestVoidCallbackFunction* impl) { return impl ? toJS(*impl) : JSC::jsNull(); }

template<> struct JSDOMCallbackConverterTraits<JSTestVoidCallbackFunction> {
    using Base = TestVoidCallbackFunction;
};
} // namespace WebCore

#endif // ENABLE(TEST_CONDITIONAL)
