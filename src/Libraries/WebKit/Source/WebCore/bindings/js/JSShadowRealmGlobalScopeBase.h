/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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

#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMWrapper.h"

namespace WebCore {

class ShadowRealmGlobalScope;

class JSShadowRealmGlobalScopeBase : public JSDOMGlobalObject {
public:
    using Base = JSDOMGlobalObject;

    static void destroy(JSC::JSCell*);

    DECLARE_INFO;

    ShadowRealmGlobalScope& wrapped() const { return *m_wrapped; }

    const JSDOMGlobalObject* incubatingRealm() const;
    JSDOMGlobalObject* incubatingRealm();

    ScriptExecutionContext* scriptExecutionContext() const;

private:
    static bool supportsRichSourceInfo(const JSC::JSGlobalObject*);
    static bool shouldInterruptScript(const JSC::JSGlobalObject*);
    static bool shouldInterruptScriptBeforeTimeout(const JSC::JSGlobalObject*);
    static JSC::RuntimeFlags javaScriptRuntimeFlags(const JSC::JSGlobalObject*);
    static JSC::ScriptExecutionStatus scriptExecutionStatus(JSC::JSGlobalObject*, JSC::JSObject*);
    static void queueMicrotaskToEventLoop(JSC::JSGlobalObject&, JSC::QueuedTask&&);
    static void reportViolationForUnsafeEval(JSC::JSGlobalObject*, const String&);

protected:
    JSShadowRealmGlobalScopeBase(JSC::VM&, JSC::Structure*, RefPtr<ShadowRealmGlobalScope>&&);
    void finishCreation(JSC::VM&, JSC::JSGlobalProxy*);

    DECLARE_VISIT_CHILDREN;

    static const JSC::GlobalObjectMethodTable* globalObjectMethodTable();

private:
    RefPtr<ShadowRealmGlobalScope> m_wrapped;
};

inline JSDOMGlobalObject* JSShadowRealmGlobalScopeBase::incubatingRealm()
{
    return const_cast<JSDOMGlobalObject*>(const_cast<const JSShadowRealmGlobalScopeBase*>(this)->incubatingRealm());
}

// Always ignores the execState and passed globalObject, ShadowRealmGlobalScope is itself a globalObject and will always use its own prototype chain.
JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, ShadowRealmGlobalScope&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, ShadowRealmGlobalScope* scope) { return scope ? toJS(lexicalGlobalObject, globalObject, *scope) : JSC::jsNull(); }
JSC::JSValue toJS(JSC::JSGlobalObject*, ShadowRealmGlobalScope&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, ShadowRealmGlobalScope* scope) { return scope ? toJS(lexicalGlobalObject, *scope) : JSC::jsNull(); }

} // namespace WebCore
