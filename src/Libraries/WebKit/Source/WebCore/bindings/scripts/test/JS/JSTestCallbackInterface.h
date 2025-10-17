/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include "JSDOMConvertDictionary.h"
#include "JSDOMConvertEnumeration.h"
#include "TestCallbackInterface.h"
#include <wtf/Forward.h>

namespace WebCore {

class JSTestCallbackInterface final : public TestCallbackInterface {
public:
    static Ref<JSTestCallbackInterface> create(JSC::JSObject* callback, JSDOMGlobalObject* globalObject)
    {
        return adoptRef(*new JSTestCallbackInterface(callback, globalObject));
    }

    ScriptExecutionContext* scriptExecutionContext() const { return ContextDestructionObserver::scriptExecutionContext(); }

    ~JSTestCallbackInterface() final;
    JSCallbackData* callbackData() { return m_data; }
    static JSC::JSValue getConstructor(JSC::VM&, const JSC::JSGlobalObject*);

    // Functions
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithNoParam() override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithNoParamRethrowingException() override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithArrayParam(typename IDLFloat32Array::ParameterType arrayParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithArrayParamRethrowingException(typename IDLFloat32Array::ParameterType arrayParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithSerializedScriptValueParam(typename IDLSerializedScriptValue<SerializedScriptValue>::ParameterType srzParam, typename IDLDOMString::ParameterType strParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithSerializedScriptValueParamRethrowingException(typename IDLSerializedScriptValue<SerializedScriptValue>::ParameterType srzParam, typename IDLDOMString::ParameterType strParam) override;
    CallbackResult<typename IDLLong::CallbackReturnType> customCallback(typename IDLInterface<TestObj>::ParameterType testObjParam, typename IDLInterface<TestNode>::ParameterType testNodeParam) override;
    CallbackResult<typename IDLLong::CallbackReturnType> customCallbackRethrowingException(typename IDLInterface<TestObj>::ParameterType testObjParam, typename IDLInterface<TestNode>::ParameterType testNodeParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithStringList(typename IDLInterface<DOMStringList>::ParameterType listParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithStringListRethrowingException(typename IDLInterface<DOMStringList>::ParameterType listParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithBoolean(typename IDLBoolean::ParameterType boolParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackWithBooleanRethrowingException(typename IDLBoolean::ParameterType boolParam) override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> callbackWithEnum(typename IDLEnumeration<TestCallbackInterface::Enum>::ParameterType enumParam) override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> callbackWithEnumRethrowingException(typename IDLEnumeration<TestCallbackInterface::Enum>::ParameterType enumParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackRequiresThisToPass(typename IDLLong::ParameterType longParam, typename IDLInterface<TestNode>::ParameterType testNodeParam) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> callbackRequiresThisToPassRethrowingException(typename IDLLong::ParameterType longParam, typename IDLInterface<TestNode>::ParameterType testNodeParam) override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> callbackWithAReturnValue() override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> callbackWithAReturnValueRethrowingException() override;
    CallbackResult<typename IDLPromise<IDLUndefined>::CallbackReturnType> callbackThatTreatsExceptionAsRejectedPromise(typename IDLEnumeration<TestCallbackInterface::Enum>::ParameterType enumParam) override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> callbackWithThisObject(typename IDLInterface<TestNode>::ParameterType thisObject, typename IDLInterface<TestObj>::ParameterType testObjParam) override;
    CallbackResult<typename IDLDOMString::CallbackReturnType> callbackWithThisObjectRethrowingException(typename IDLInterface<TestNode>::ParameterType thisObject, typename IDLInterface<TestObj>::ParameterType testObjParam) override;

private:
    JSTestCallbackInterface(JSC::JSObject*, JSDOMGlobalObject*);

    bool hasCallback() const final { return m_data && m_data->callback(); }

    void visitJSFunction(JSC::AbstractSlotVisitor&) override;

    void visitJSFunction(JSC::SlotVisitor&) override;

    JSCallbackData* m_data;
};

JSC::JSValue toJS(TestCallbackInterface&);
inline JSC::JSValue toJS(TestCallbackInterface* impl) { return impl ? toJS(*impl) : JSC::jsNull(); }

template<> struct JSDOMCallbackConverterTraits<JSTestCallbackInterface> {
    using Base = TestCallbackInterface;
};
String convertEnumerationToString(TestCallbackInterface::Enum);
template<> JSC::JSString* convertEnumerationToJS(JSC::VM&, TestCallbackInterface::Enum);

template<> std::optional<TestCallbackInterface::Enum> parseEnumerationFromString<TestCallbackInterface::Enum>(const String&);
template<> std::optional<TestCallbackInterface::Enum> parseEnumeration<TestCallbackInterface::Enum>(JSC::JSGlobalObject&, JSC::JSValue);
template<> ASCIILiteral expectedEnumerationValues<TestCallbackInterface::Enum>();

template<> ConversionResult<IDLDictionary<TestCallbackInterface::Dictionary>> convertDictionary<TestCallbackInterface::Dictionary>(JSC::JSGlobalObject&, JSC::JSValue);

} // namespace WebCore

#endif // ENABLE(TEST_CONDITIONAL)
