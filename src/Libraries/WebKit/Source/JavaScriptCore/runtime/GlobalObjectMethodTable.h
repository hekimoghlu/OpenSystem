/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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

#include <wtf/Forward.h>

namespace JSC {

class Identifier;
class JSGlobalObject;
class JSInternalPromise;
class JSModuleLoader;
class JSModuleRecord;
class JSObject;
class JSPromise;
class JSString;
class JSValue;
class Microtask;
class RuntimeFlags;
class SourceOrigin;
class Structure;
class QueuedTask;

enum class CompilationType;
enum class ScriptExecutionStatus;

enum class JSPromiseRejectionOperation : unsigned {
    Reject, // When a promise is rejected without any handlers.
    Handle, // When a handler is added to a rejected promise for the first time.
};

struct GlobalObjectMethodTable {
    bool (*supportsRichSourceInfo)(const JSGlobalObject*);
    bool (*shouldInterruptScript)(const JSGlobalObject*);
    RuntimeFlags (*javaScriptRuntimeFlags)(const JSGlobalObject*);
    void (*queueMicrotaskToEventLoop)(JSGlobalObject&, QueuedTask&&);
    bool (*shouldInterruptScriptBeforeTimeout)(const JSGlobalObject*);

    JSInternalPromise* (*moduleLoaderImportModule)(JSGlobalObject*, JSModuleLoader*, JSString*, JSValue, const SourceOrigin&);
    Identifier (*moduleLoaderResolve)(JSGlobalObject*, JSModuleLoader*, JSValue, JSValue, JSValue);
    JSInternalPromise* (*moduleLoaderFetch)(JSGlobalObject*, JSModuleLoader*, JSValue, JSValue, JSValue);
    JSObject* (*moduleLoaderCreateImportMetaProperties)(JSGlobalObject*, JSModuleLoader*, JSValue, JSModuleRecord*, JSValue);
    JSValue (*moduleLoaderEvaluate)(JSGlobalObject*, JSModuleLoader*, JSValue key, JSValue moduleRecordValue, JSValue scriptFetcher, JSValue awaitedValue, JSValue resumeMode);

    void (*promiseRejectionTracker)(JSGlobalObject*, JSPromise*, JSPromiseRejectionOperation);
    void (*reportUncaughtExceptionAtEventLoop)(JSGlobalObject*, Exception*);

    // For most contexts this is just the global object. For JSDOMWindow, however, this is the JSDocument.
    JSObject* (*currentScriptExecutionOwner)(JSGlobalObject*);

    ScriptExecutionStatus (*scriptExecutionStatus)(JSGlobalObject*, JSObject* scriptExecutionOwner);
    void (*reportViolationForUnsafeEval)(JSGlobalObject*, const String&);
    String (*defaultLanguage)();
    JSPromise* (*compileStreaming)(JSGlobalObject*, JSValue);
    JSPromise* (*instantiateStreaming)(JSGlobalObject*, JSValue, JSObject*);
    JSGlobalObject* (*deriveShadowRealmGlobalObject)(JSGlobalObject*);
    String (*codeForEval)(JSGlobalObject*, JSValue);
    bool (*canCompileStrings)(JSGlobalObject*, CompilationType, String, const ArgList&);
    Structure* (*trustedScriptStructure)(JSGlobalObject*);
};

} // namespace JSC
