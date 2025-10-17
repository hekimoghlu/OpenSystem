/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include "Exception.h"
#include "InspectorEnvironment.h"
#include "InspectorProtocolObjects.h"
#include "ScriptFunctionCall.h"
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/NakedPtr.h>
#include <wtf/RefPtr.h>

namespace Deprecated {
class ScriptFunctionCall;
}

namespace Inspector {

using AsyncCallCallback = WTF::Function<void(Protocol::ErrorString&, RefPtr<Protocol::Runtime::RemoteObject>&&, std::optional<bool>&&, std::optional<int>&&)>;

JS_EXPORT_PRIVATE RefPtr<JSON::Value> toInspectorValue(JSC::JSGlobalObject*, JSC::JSValue);

class InjectedScriptBase {
public:
    JS_EXPORT_PRIVATE InjectedScriptBase(const InjectedScriptBase&);
    JS_EXPORT_PRIVATE virtual ~InjectedScriptBase();

    JS_EXPORT_PRIVATE InjectedScriptBase& operator=(const InjectedScriptBase&);

    const String& name() const { return m_name; }
    bool hasNoValue() const { return !m_injectedScriptObject.get(); }
    JSC::JSGlobalObject* globalObject() const { return m_globalObject; }

protected:
    InjectedScriptBase(const String& name);
    InjectedScriptBase(const String& name, JSC::JSGlobalObject*, JSC::JSObject*, InspectorEnvironment*);

    InspectorEnvironment* inspectorEnvironment() const { return m_environment; }

    bool hasAccessToInspectedScriptState() const;

    JSC::JSObject* injectedScriptObject() const;
    Expected<JSC::JSValue, NakedPtr<JSC::Exception>> callFunctionWithEvalEnabled(ScriptFunctionCall&) const;
    Ref<JSON::Value> makeCall(ScriptFunctionCall&);
    void makeEvalCall(Protocol::ErrorString&, ScriptFunctionCall&, RefPtr<Protocol::Runtime::RemoteObject>& resultObject, std::optional<bool>& wasThrown, std::optional<int>& savedResultIndex);
    void makeAsyncCall(ScriptFunctionCall&, AsyncCallCallback&&);

private:
    void checkCallResult(Protocol::ErrorString&, RefPtr<JSON::Value> result, RefPtr<Protocol::Runtime::RemoteObject>& resultObject, std::optional<bool>& wasThrown, std::optional<int>& savedResultIndex);
    void checkAsyncCallResult(RefPtr<JSON::Value> result, const AsyncCallCallback&);

    String m_name;
    JSC::JSGlobalObject* m_globalObject { nullptr };
    JSC::Strong<JSC::JSObject> m_injectedScriptObject;
    InspectorEnvironment* m_environment { nullptr };
};

} // namespace Inspector
