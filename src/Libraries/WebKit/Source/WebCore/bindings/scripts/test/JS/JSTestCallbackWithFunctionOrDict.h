/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "TestCallbackWithFunctionOrDict.h"
#include <wtf/Forward.h>

namespace WebCore {

class JSTestCallbackWithFunctionOrDict final : public TestCallbackWithFunctionOrDict {
public:
    static Ref<JSTestCallbackWithFunctionOrDict> create(JSC::JSObject* callback, JSDOMGlobalObject* globalObject)
    {
        return adoptRef(*new JSTestCallbackWithFunctionOrDict(callback, globalObject));
    }

    ScriptExecutionContext* scriptExecutionContext() const { return ContextDestructionObserver::scriptExecutionContext(); }

    ~JSTestCallbackWithFunctionOrDict() final;
    JSCallbackData* callbackData() { return m_data; }

    // Functions
    CallbackResult<typename IDLUndefined::CallbackReturnType> handleEvent(typename IDLUnion<IDLDictionary<TestDictionary>, IDLCallbackFunction<JSTestCallbackFunction>>::ParameterType callback) override;
    CallbackResult<typename IDLUndefined::CallbackReturnType> handleEventRethrowingException(typename IDLUnion<IDLDictionary<TestDictionary>, IDLCallbackFunction<JSTestCallbackFunction>>::ParameterType callback) override;

private:
    JSTestCallbackWithFunctionOrDict(JSC::JSObject*, JSDOMGlobalObject*);

    bool hasCallback() const final { return m_data && m_data->callback(); }

    void visitJSFunction(JSC::AbstractSlotVisitor&) override;

    void visitJSFunction(JSC::SlotVisitor&) override;

    JSCallbackData* m_data;
};

JSC::JSValue toJS(TestCallbackWithFunctionOrDict&);
inline JSC::JSValue toJS(TestCallbackWithFunctionOrDict* impl) { return impl ? toJS(*impl) : JSC::jsNull(); }

template<> struct JSDOMCallbackConverterTraits<JSTestCallbackWithFunctionOrDict> {
    using Base = TestCallbackWithFunctionOrDict;
};
} // namespace WebCore
