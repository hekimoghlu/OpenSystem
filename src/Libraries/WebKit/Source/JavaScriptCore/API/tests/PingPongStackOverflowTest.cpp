/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "PingPongStackOverflowTest.h"

#include "InitializeThreading.h"
#include "JavaScript.h"
#include "Options.h"
#include <wtf/text/StringBuilder.h>

using JSC::Options;

static JSGlobalContextRef context = nullptr;
static int nativeRecursionCount = 0;

static bool PingPongStackOverflowObject_hasInstance(JSContextRef context, JSObjectRef constructor, JSValueRef possibleValue, JSValueRef* exception)
{
    UNUSED_PARAM(context);
    UNUSED_PARAM(constructor);

    JSStringRef hasInstanceName = JSStringCreateWithUTF8CString("hasInstance");
    JSValueRef hasInstance = JSObjectGetProperty(context, constructor, hasInstanceName, exception);
    JSStringRelease(hasInstanceName);
    if (!hasInstance)
        return false;

    int countAtEntry = nativeRecursionCount++;

    JSValueRef result = nullptr;
    if (nativeRecursionCount < 100) {
        JSObjectRef function = JSValueToObject(context, hasInstance, exception);
        result = JSObjectCallAsFunction(context, function, constructor, 1, &possibleValue, exception);
    } else {
        StringBuilder builder;
        builder.append("dummy.valueOf([0]"_s);
        for (int i = 1; i < 35000; i++)
            builder.append(", ["_s, i, ']');
        builder.append(");"_s);
        JSStringRef script = JSStringCreateWithUTF8CString(builder.toString().utf8().data());
        result = JSEvaluateScript(context, script, nullptr, nullptr, 1, exception);
        JSStringRelease(script);
    }

    --nativeRecursionCount;
    if (nativeRecursionCount != countAtEntry)
        printf("    ERROR: PingPongStackOverflow test saw a recursion count mismatch\n");

    return result && JSValueToBoolean(context, result);
}

static const JSClassDefinition PingPongStackOverflowObject_definition = {
    0,
    kJSClassAttributeNone,
    
    "PingPongStackOverflowObject",
    nullptr,
    
    nullptr,
    nullptr,
    
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    PingPongStackOverflowObject_hasInstance,
    nullptr,
};

static JSClassRef PingPongStackOverflowObject_class(JSContextRef context)
{
    UNUSED_PARAM(context);
    
    static JSClassRef jsClass;
    if (!jsClass)
        jsClass = JSClassCreate(&PingPongStackOverflowObject_definition);
    
    return jsClass;
}

// This tests tests a stack overflow on VM reentry into a JS function from a native function
// after ping-pong'ing back and forth between JS and native functions multiple times.
// This test should not hang or crash.
int testPingPongStackOverflow()
{
    bool failed = false;

    JSC::initialize();

    auto origSoftReservedZoneSize = Options::softReservedZoneSize();
    auto origReservedZoneSize = Options::reservedZoneSize();
    auto origUseLLInt = Options::useLLInt();
    auto origMaxPerThreadStackUsage = Options::maxPerThreadStackUsage();

    Options::softReservedZoneSize() = 128 * KB;
    Options::reservedZoneSize() = 64 * KB;
#if ENABLE(JIT)
    // Normally, we want to disable the LLINT to force the use of JITted code which is necessary for
    // reproducing the regression in https://bugs.webkit.org/show_bug.cgi?id=148749. However, we only
    // want to do this if the LLINT isn't the only available execution engine.
    if (Options::useJIT())
        Options::useLLInt() = false;
#endif

    const char* scriptString =
        "var count = 0;" \
        "PingPongStackOverflowObject.hasInstance = function f() {" \
        "    return (undefined instanceof PingPongStackOverflowObject);" \
        "};" \
        "PingPongStackOverflowObject.__proto__ = undefined;" \
        "undefined instanceof PingPongStackOverflowObject;";

    JSValueRef exception = nullptr;
    JSStringRef script = JSStringCreateWithUTF8CString(scriptString);

    nativeRecursionCount = 0;
    context = JSGlobalContextCreateInGroup(nullptr, nullptr);

    JSObjectRef globalObject = JSContextGetGlobalObject(context);
    ASSERT(JSValueIsObject(context, globalObject));

    JSObjectRef PingPongStackOverflowObject = JSObjectMake(context, PingPongStackOverflowObject_class(context), nullptr);
    JSStringRef PingPongStackOverflowObjectString = JSStringCreateWithUTF8CString("PingPongStackOverflowObject");
    JSObjectSetProperty(context, globalObject, PingPongStackOverflowObjectString, PingPongStackOverflowObject, kJSPropertyAttributeNone, nullptr);
    JSStringRelease(PingPongStackOverflowObjectString);

    unsigned stackSize = 32 * KB;
    Options::maxPerThreadStackUsage() = stackSize + Options::softReservedZoneSize();

    exception = nullptr;
    JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);

    JSGlobalContextRelease(context);
    context = nullptr;
    JSStringRelease(script);

    if (!exception) {
        printf("FAIL: PingPongStackOverflowError not thrown in PingPongStackOverflow test\n");
        failed = true;
    } else if (nativeRecursionCount) {
        printf("FAIL: Unbalanced native recursion count: %d in PingPongStackOverflow test\n", nativeRecursionCount);
        failed = true;
    } else {
        printf("PASS: PingPongStackOverflow test.\n");
    }

    Options::softReservedZoneSize() = origSoftReservedZoneSize;
    Options::reservedZoneSize() = origReservedZoneSize;
    Options::useLLInt() = origUseLLInt;
    Options::maxPerThreadStackUsage() = origMaxPerThreadStackUsage;

    return failed;
}
