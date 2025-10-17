/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "JSBase.h"
#include "JSBaseInternal.h"
#include "JSBasePrivate.h"

#include "APICast.h"
#include "Completion.h"
#include "GCActivityCallback.h"
#include "JSCInlines.h"
#include "JSLock.h"
#include "ObjectConstructor.h"
#include "OpaqueJSString.h"
#include "SourceCode.h"

#if ENABLE(REMOTE_INSPECTOR)
#include "JSGlobalObjectInspectorController.h"
#endif

using namespace JSC;

JSValueRef JSEvaluateScriptInternal(const JSLockHolder&, JSContextRef ctx, JSObjectRef thisObject, const SourceCode& source, JSValueRef* exception)
{
    JSObject* jsThisObject = toJS(thisObject);

    // evaluate sets "this" to the global object if it is NULL
    JSGlobalObject* globalObject = toJS(ctx);
    NakedPtr<Exception> evaluationException;
    JSValue returnValue = profiledEvaluate(globalObject, ProfilingReason::API, source, jsThisObject, evaluationException);

    if (evaluationException) {
        if (exception)
            *exception = toRef(globalObject, evaluationException->value());
#if ENABLE(REMOTE_INSPECTOR)
        // FIXME: If we have a debugger attached we could learn about ParseError exceptions through
        // ScriptDebugServer::sourceParsed and this path could produce a duplicate warning. The
        // Debugger path is currently ignored by inspector.
        // NOTE: If we don't have a debugger, this SourceCode will be forever lost to the inspector.
        // We could stash it in the inspector in case an inspector is ever opened.
        globalObject->inspectorController().reportAPIException(globalObject, evaluationException);
#endif
        return nullptr;
    }

    if (returnValue)
        return toRef(globalObject, returnValue);

    // happens, for example, when the only statement is an empty (';') statement
    return toRef(globalObject, jsUndefined());
}

JSValueRef JSEvaluateScript(JSContextRef ctx, JSStringRef script, JSObjectRef thisObject, JSStringRef sourceURLString, int startingLineNumber, JSValueRef* exception)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);

    startingLineNumber = std::max(1, startingLineNumber);

    auto sourceURL = sourceURLString ? URL({ }, sourceURLString->string()) : URL();
    SourceCode source = makeSource(script->string(), SourceOrigin { sourceURL }, SourceTaintedOrigin::Untainted, sourceURL.string(), TextPosition(OrdinalNumber::fromOneBasedInt(startingLineNumber), OrdinalNumber()));

    return JSEvaluateScriptInternal(locker, ctx, thisObject, source, exception);
}

bool JSCheckScriptSyntax(JSContextRef ctx, JSStringRef script, JSStringRef sourceURLString, int startingLineNumber, JSValueRef* exception)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return false;
    }
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);

    startingLineNumber = std::max(1, startingLineNumber);

    auto sourceURL = sourceURLString ? URL({ }, sourceURLString->string()) : URL();
    SourceCode source = makeSource(script->string(), SourceOrigin { sourceURL }, SourceTaintedOrigin::Untainted, sourceURL.string(), TextPosition(OrdinalNumber::fromOneBasedInt(startingLineNumber), OrdinalNumber()));
    
    JSValue syntaxException;
    bool isValidSyntax = checkSyntax(globalObject, source, &syntaxException);

    if (!isValidSyntax) {
        if (exception)
            *exception = toRef(globalObject, syntaxException);
#if ENABLE(REMOTE_INSPECTOR)
        Exception* exception = Exception::create(vm, syntaxException);
        globalObject->inspectorController().reportAPIException(globalObject, exception);
#endif
        return false;
    }

    return true;
}

void JSGarbageCollect(JSContextRef ctx)
{
    // We used to recommend passing NULL as an argument here, which caused the only heap to be collected.
    // As there is no longer a shared heap, the previously recommended usage became a no-op (but the GC
    // will happen when the context group is destroyed).
    // Because the function argument was originally ignored, some clients may pass their released context here,
    // in which case there is a risk of crashing if another thread performs GC on the same heap in between.
    if (!ctx)
        return;

    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);

    vm.heap.reportAbandonedObjectGraph();
}

void JSReportExtraMemoryCost(JSContextRef ctx, size_t size)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return;
    }
    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);

    vm.heap.deprecatedReportExtraMemory(size);
}

extern "C" JS_EXPORT void JSSynchronousGarbageCollectForDebugging(JSContextRef);
extern "C" JS_EXPORT void JSSynchronousEdenCollectForDebugging(JSContextRef);

void JSSynchronousGarbageCollectForDebugging(JSContextRef ctx)
{
    if (!ctx)
        return;

    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);
    vm.heap.collectNow(Sync, CollectionScope::Full);
}

void JSSynchronousEdenCollectForDebugging(JSContextRef ctx)
{
    if (!ctx)
        return;

    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);
    vm.heap.collectSync(CollectionScope::Eden);
}

void JSDisableGCTimer(void)
{
    GCActivityCallback::s_shouldCreateGCTimer = false;
}

#if !OS(DARWIN) && !OS(WINDOWS)
bool JSConfigureSignalForGC(int signal)
{
    if (g_wtfConfig.isThreadSuspendResumeSignalConfigured)
        return false;
    g_wtfConfig.sigThreadSuspendResume = signal;
    g_wtfConfig.isUserSpecifiedThreadSuspendResumeSignalConfigured = true;
    return true;
}
#endif

JSObjectRef JSGetMemoryUsageStatistics(JSContextRef ctx)
{
    if (!ctx) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    JSGlobalObject* globalObject = toJS(ctx);
    VM& vm = globalObject->vm();
    JSLockHolder locker(vm);

    auto typeCounts = vm.heap.objectTypeCounts();
    JSObject* objectTypeCounts = constructEmptyObject(globalObject);
    for (auto& it : *typeCounts)
        objectTypeCounts->putDirect(vm, Identifier::fromLatin1(vm, it.key), jsNumber(it.value));

    JSObject* object = constructEmptyObject(globalObject);
    object->putDirect(vm, Identifier::fromString(vm, "heapSize"_s), jsNumber(vm.heap.size()));
    object->putDirect(vm, Identifier::fromString(vm, "heapCapacity"_s), jsNumber(vm.heap.capacity()));
    object->putDirect(vm, Identifier::fromString(vm, "extraMemorySize"_s), jsNumber(vm.heap.extraMemorySize()));
    object->putDirect(vm, Identifier::fromString(vm, "objectCount"_s), jsNumber(vm.heap.objectCount()));
    object->putDirect(vm, Identifier::fromString(vm, "protectedObjectCount"_s), jsNumber(vm.heap.protectedObjectCount()));
    object->putDirect(vm, Identifier::fromString(vm, "globalObjectCount"_s), jsNumber(vm.heap.globalObjectCount()));
    object->putDirect(vm, Identifier::fromString(vm, "protectedGlobalObjectCount"_s), jsNumber(vm.heap.protectedGlobalObjectCount()));
    object->putDirect(vm, Identifier::fromString(vm, "objectTypeCounts"_s), objectTypeCounts);

    return toRef(object);
}
