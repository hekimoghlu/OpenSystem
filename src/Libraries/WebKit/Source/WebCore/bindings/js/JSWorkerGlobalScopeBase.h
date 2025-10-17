/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#include "JSDOMGlobalObject.h"
#include "JSDOMWrapper.h"
#include "ServiceWorkerGlobalScope.h"

namespace WebCore {

class JSWorkerGlobalScopeBase : public JSDOMGlobalObject {
public:
    using Base = JSDOMGlobalObject;

    template<typename, JSC::SubspaceAccess>
    static void subspaceFor(JSC::VM&) { RELEASE_ASSERT_NOT_REACHED(); }

    static void destroy(JSC::JSCell*);

    DECLARE_INFO;

    WorkerGlobalScope& wrapped() const { return *m_wrapped; }
    ScriptExecutionContext* scriptExecutionContext() const;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::GlobalObjectType, StructureFlags), info());
    }

    static bool supportsRichSourceInfo(const JSC::JSGlobalObject*);
    static bool shouldInterruptScript(const JSC::JSGlobalObject*);
    static bool shouldInterruptScriptBeforeTimeout(const JSC::JSGlobalObject*);
    static JSC::RuntimeFlags javaScriptRuntimeFlags(const JSC::JSGlobalObject*);
    static JSC::ScriptExecutionStatus scriptExecutionStatus(JSC::JSGlobalObject*, JSC::JSObject*);
    static void queueMicrotaskToEventLoop(JSC::JSGlobalObject&, JSC::QueuedTask&&);
    static void reportViolationForUnsafeEval(JSC::JSGlobalObject*, const String&);

protected:
    JSWorkerGlobalScopeBase(JSC::VM&, JSC::Structure*, RefPtr<WorkerGlobalScope>&&);
    void finishCreation(JSC::VM&, JSC::JSGlobalProxy*);

    DECLARE_VISIT_CHILDREN;

    static const JSC::GlobalObjectMethodTable* globalObjectMethodTable();

private:
    RefPtr<WorkerGlobalScope> m_wrapped;
};

// Returns a JSWorkerGlobalScope or jsNull()
// Always ignores the execState and passed globalObject, WorkerGlobalScope is itself a globalObject and will always use its own prototype chain.
JSC::JSValue toJS(JSC::JSGlobalObject*, JSDOMGlobalObject*, WorkerGlobalScope&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, WorkerGlobalScope* scope) { return scope ? toJS(lexicalGlobalObject, globalObject, *scope) : JSC::jsNull(); }
JSC::JSValue toJS(JSC::JSGlobalObject*, WorkerGlobalScope&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, WorkerGlobalScope* scope) { return scope ? toJS(lexicalGlobalObject, *scope) : JSC::jsNull(); }

} // namespace WebCore
