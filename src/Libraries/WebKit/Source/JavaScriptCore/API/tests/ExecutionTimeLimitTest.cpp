/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include "ExecutionTimeLimitTest.h"

#include "InitializeThreading.h"
#include "JSContextRefPrivate.h"
#include "JavaScript.h"
#include "Options.h"
#include <wtf/CPUTime.h>
#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/Threading.h>
#include <wtf/WTFProcess.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

#if HAVE(MACH_EXCEPTIONS)
#include <dispatch/dispatch.h>
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

using JSC::Options;

static JSGlobalContextRef context = nullptr;

static JSValueRef currentCPUTimeAsJSFunctionCallback(JSContextRef ctx, JSObjectRef functionObject, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(functionObject);
    UNUSED_PARAM(thisObject);
    UNUSED_PARAM(argumentCount);
    UNUSED_PARAM(arguments);
    UNUSED_PARAM(exception);
    
    ASSERT(JSContextGetGlobalContext(ctx) == context);
    return JSValueMakeNumber(ctx, CPUTime::forCurrentThread().seconds());
}

bool shouldTerminateCallbackWasCalled = false;
static bool shouldTerminateCallback(JSContextRef, void*)
{
    shouldTerminateCallbackWasCalled = true;
    return true;
}

bool cancelTerminateCallbackWasCalled = false;
static bool cancelTerminateCallback(JSContextRef, void*)
{
    cancelTerminateCallbackWasCalled = true;
    return false;
}

int extendTerminateCallbackCalled = 0;
static bool extendTerminateCallback(JSContextRef ctx, void*)
{
    extendTerminateCallbackCalled++;
    if (extendTerminateCallbackCalled == 1) {
        JSContextGroupRef contextGroup = JSContextGetGroup(ctx);
        JSContextGroupSetExecutionTimeLimit(contextGroup, .200f, extendTerminateCallback, nullptr);
        return false;
    }
    return true;
}

#if HAVE(MACH_EXCEPTIONS)
bool dispatchTerminateCallbackCalled = false;
static bool dispatchTermitateCallback(JSContextRef, void*)
{
    dispatchTerminateCallbackCalled = true;
    return true;
}
#endif

enum class Tier {
    LLInt,
    Baseline,
    DFG,
    FTL
};

struct TierOptions {
    Tier tier;
    Seconds timeLimitAdjustment;
    const char* optionsStr;
};

static void testResetAfterTimeout(bool& failed)
{
    JSValueRef v = nullptr;
    JSValueRef exception = nullptr;
    const char* reentryScript = "100";
    JSStringRef script = JSStringCreateWithUTF8CString(reentryScript);
    v = JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
    JSStringRelease(script);
    if (exception) {
        printf("FAIL: Watchdog timeout was not reset.\n");
        failed = true;
    } else if (!JSValueIsNumber(context, v) || JSValueToNumber(context, v, nullptr) != 100) {
        printf("FAIL: Script result is not as expected.\n");
        failed = true;
    }
}

int testExecutionTimeLimit()
{
    static const TierOptions tierOptionsList[] = {
        { Tier::LLInt,    0_ms,   "--useConcurrentJIT=false --useLLInt=true --useBaselineJIT=false" },
#if ENABLE(JIT)
        { Tier::Baseline, 0_ms,   "--useConcurrentJIT=false --useLLInt=true --useBaselineJIT=true --useDFGJIT=false" },
        { Tier::DFG,      200_ms,   "--useConcurrentJIT=false --useLLInt=true --useBaselineJIT=true --useDFGJIT=true --useFTLJIT=false" },
#if ENABLE(FTL_JIT)
        { Tier::FTL,      500_ms, "--useConcurrentJIT=false --useLLInt=true --useBaselineJIT=true --useDFGJIT=true --useFTLJIT=true" },
#endif
#endif // ENABLE(JIT)
    };

    auto tierNameFor = [] (Tier tier) -> const char* {
        switch (tier) {
        case Tier::LLInt:
            return "LLInt";
        case Tier::Baseline:
            return "Baseline";
        case Tier::DFG:
            return "DFG";
        case Tier::FTL:
            return "FTL";
        }
        RELEASE_ASSERT_NOT_REACHED();
        return nullptr;
    };

    bool failed = false;

    JSC::initialize();

    for (auto tierOptions : tierOptionsList) {
        if (!Options::useJIT() && tierOptions.tier > Tier::LLInt)
            break;

        const char* tierName = tierNameFor(tierOptions.tier);
        StringBuilder savedOptionsBuilder;
        Options::dumpAllOptionsInALine(savedOptionsBuilder);

        Options::setOptions(tierOptions.optionsStr);
        
        Seconds tierAdjustment = tierOptions.timeLimitAdjustment;
        Seconds timeLimit;

        context = JSGlobalContextCreateInGroup(nullptr, nullptr);

        JSContextGroupRef contextGroup = JSContextGetGroup(context);
        JSObjectRef globalObject = JSContextGetGlobalObject(context);
        ASSERT(JSValueIsObject(context, globalObject));

        JSValueRef exception = nullptr;

        JSStringRef currentCPUTimeStr = JSStringCreateWithUTF8CString("currentCPUTime");
        JSObjectRef currentCPUTimeFunction = JSObjectMakeFunctionWithCallback(context, currentCPUTimeStr, currentCPUTimeAsJSFunctionCallback);
        JSObjectSetProperty(context, globalObject, currentCPUTimeStr, currentCPUTimeFunction, kJSPropertyAttributeNone, nullptr);
        JSStringRelease(currentCPUTimeStr);

        /* Test script on another thread: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), shouldTerminateCallback, nullptr);
        {
#if OS(LINUX) && CPU(ARM_THUMB2)
            Seconds timeAfterWatchdogShouldHaveFired = 500_ms + tierAdjustment;
#else
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;
#endif

            JSStringRef script = JSStringCreateWithUTF8CString("function foo() { while (true) { } } foo();");
            exception = nullptr;
            JSValueRef* exn = &exception;
            shouldTerminateCallbackWasCalled = false;
            auto thread = Thread::create("Rogue thread"_s, [=] {
                JSEvaluateScript(context, script, nullptr, nullptr, 1, exn);
            });

            sleep(timeAfterWatchdogShouldHaveFired);

            if (shouldTerminateCallbackWasCalled)
                printf("PASS: %s script timed out as expected.\n", tierName);
            else {
                printf("FAIL: %s script timeout callback was not called.\n", tierName);
                exitProcess(1);
            }

            if (!exception) {
                printf("FAIL: %s TerminationException was not thrown.\n", tierName);
                exitProcess(1);
            }

            thread->waitForCompletion();
            testResetAfterTimeout(failed);

            JSStringRelease(script);
        }

        /* Test script timeout: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), shouldTerminateCallback, nullptr);
        {
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;

            CString scriptText = makeString(
                "function foo() {"
                    "var startTime = currentCPUTime();"
                    "while (true) {"
                        "for (var i = 0; i < 1000; i++);"
                        "if (currentCPUTime() - startTime > "_s, timeAfterWatchdogShouldHaveFired.seconds(), ") break;"
                    "}"
                "}"
                "foo();"_s
            ).utf8();

            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            shouldTerminateCallbackWasCalled = false;
            auto startTime = CPUTime::forCurrentThread();
            JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
            auto endTime = CPUTime::forCurrentThread();
            JSStringRelease(script);

            if (((endTime - startTime) < timeAfterWatchdogShouldHaveFired) && shouldTerminateCallbackWasCalled)
                printf("PASS: %s script timed out as expected.\n", tierName);
            else {
                if ((endTime - startTime) >= timeAfterWatchdogShouldHaveFired)
                    printf("FAIL: %s script did not time out as expected.\n", tierName);
                if (!shouldTerminateCallbackWasCalled)
                    printf("FAIL: %s script timeout callback was not called.\n", tierName);
                failed = true;
            }
            
            if (!exception) {
                printf("FAIL: %s TerminationException was not thrown.\n", tierName);
                failed = true;
            }

            testResetAfterTimeout(failed);
        }

        /* Test script timeout with tail calls: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), shouldTerminateCallback, nullptr);
        {
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;

            CString scriptText = makeString(
                "var startTime = currentCPUTime();"
                "function recurse(i) {"
                    "'use strict';"
                    "if (i % 1000 === 0) {"
                        "if (currentCPUTime() - startTime >"_s, timeAfterWatchdogShouldHaveFired.seconds(), ") { return; }"
                    "}"
                "return recurse(i + 1); }"
                "recurse(0);"_s
            ).utf8();

            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            shouldTerminateCallbackWasCalled = false;
            auto startTime = CPUTime::forCurrentThread();
            JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
            auto endTime = CPUTime::forCurrentThread();
            JSStringRelease(script);

            if (((endTime - startTime) < timeAfterWatchdogShouldHaveFired) && shouldTerminateCallbackWasCalled)
                printf("PASS: %s script with infinite tail calls timed out as expected .\n", tierName);
            else {
                if ((endTime - startTime) >= timeAfterWatchdogShouldHaveFired)
                    printf("FAIL: %s script with infinite tail calls did not time out as expected.\n", tierName);
                if (!shouldTerminateCallbackWasCalled)
                    printf("FAIL: %s script with infinite tail calls' timeout callback was not called.\n", tierName);
                failed = true;
            }
            
            if (!exception) {
                printf("FAIL: %s TerminationException was not thrown.\n", tierName);
                failed = true;
            }

            testResetAfterTimeout(failed);
        }

        /* Test the script timeout's TerminationException should NOT be catchable: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), shouldTerminateCallback, nullptr);
        {
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;
            
            CString scriptText = makeString(
                "function foo() {"
                    "var startTime = currentCPUTime();"
                    "try {"
                        "while (true) {"
                            "for (var i = 0; i < 1000; i++);"
                                "if (currentCPUTime() - startTime > "_s, timeAfterWatchdogShouldHaveFired.seconds(), ") break;"
                        "}"
                    "} catch(e) { }"
                "}"
                "foo();"_s
            ).utf8();

            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            shouldTerminateCallbackWasCalled = false;

            auto startTime = CPUTime::forCurrentThread();
            JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
            auto endTime = CPUTime::forCurrentThread();
            
            JSStringRelease(script);

            if (((endTime - startTime) >= timeAfterWatchdogShouldHaveFired) || !shouldTerminateCallbackWasCalled) {
                if (!((endTime - startTime) < timeAfterWatchdogShouldHaveFired))
                    printf("FAIL: %s script did not time out as expected.\n", tierName);
                if (!shouldTerminateCallbackWasCalled)
                    printf("FAIL: %s script timeout callback was not called.\n", tierName);
                failed = true;
            }
            
            if (exception)
                printf("PASS: %s TerminationException was not catchable as expected.\n", tierName);
            else {
                printf("FAIL: %s TerminationException was caught.\n", tierName);
                failed = true;
            }

            testResetAfterTimeout(failed);
        }
        
        /* Test script timeout with no callback: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), nullptr, nullptr);
        {
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;
            
            CString scriptText = makeString(
                "function foo() {"
                    "var startTime = currentCPUTime();"
                    "while (true) {"
                        "for (var i = 0; i < 1000; i++);"
                            "if (currentCPUTime() - startTime > "_s, timeAfterWatchdogShouldHaveFired.seconds(), ") break;"
                    "}"
                "}"
                "foo();"_s
            ).utf8();
            
            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            shouldTerminateCallbackWasCalled = false;

            auto startTime = CPUTime::forCurrentThread();
            JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
            auto endTime = CPUTime::forCurrentThread();
            
            JSStringRelease(script);

            if (((endTime - startTime) < timeAfterWatchdogShouldHaveFired) && !shouldTerminateCallbackWasCalled)
                printf("PASS: %s script timed out as expected when no callback is specified.\n", tierName);
            else {
                if ((endTime - startTime) >= timeAfterWatchdogShouldHaveFired)
                    printf("FAIL: %s script did not time out as expected when no callback is specified.\n", tierName);
                else
                    printf("FAIL: %s script called stale callback function.\n", tierName);
                failed = true;
            }
            
            if (!exception) {
                printf("FAIL: %s TerminationException was not thrown.\n", tierName);
                failed = true;
            }

            testResetAfterTimeout(failed);
        }
        
        /* Test script timeout cancellation: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), cancelTerminateCallback, nullptr);
        {
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;
            
            CString scriptText = makeString(
                "function foo() {"
                    "var startTime = currentCPUTime();"
                    "while (true) {"
                        "for (var i = 0; i < 1000; i++);"
                            "if (currentCPUTime() - startTime > "_s, timeAfterWatchdogShouldHaveFired.seconds(), ") break;"
                    "}"
                "}"
                "foo();"_s
            ).utf8();

            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            cancelTerminateCallbackWasCalled = false;

            auto startTime = CPUTime::forCurrentThread();
            JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
            auto endTime = CPUTime::forCurrentThread();
            
            JSStringRelease(script);

            if (((endTime - startTime) >= timeAfterWatchdogShouldHaveFired) && cancelTerminateCallbackWasCalled && !exception)
                printf("PASS: %s script timeout was cancelled as expected.\n", tierName);
            else {
                if (((endTime - startTime) < timeAfterWatchdogShouldHaveFired) || exception)
                    printf("FAIL: %s script timeout was not cancelled.\n", tierName);
                if (!cancelTerminateCallbackWasCalled)
                    printf("FAIL: %s script timeout callback was not called.\n", tierName);
                failed = true;
            }
            
            if (exception) {
                printf("FAIL: %s Unexpected TerminationException thrown.\n", tierName);
                failed = true;
            }
        }
        
        /* Test script timeout extension: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), extendTerminateCallback, nullptr);
        {
            Seconds timeBeforeExtendedDeadline = 250_ms + tierAdjustment;
            Seconds timeAfterExtendedDeadline = 600_ms + tierAdjustment;
            Seconds maxBusyLoopTime = 750_ms + tierAdjustment;

            CString scriptText = makeString(
                "function foo() {"
                    "var startTime = currentCPUTime();"
                    "while (true) {"
                        "for (var i = 0; i < 1000; i++);"
                            "if (currentCPUTime() - startTime > "_s, maxBusyLoopTime.seconds(), ") break;"
                    "}"
                "}"
                "foo();"_s
            ).utf8();

            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            extendTerminateCallbackCalled = 0;

            auto startTime = CPUTime::forCurrentThread();
            JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
            auto endTime = CPUTime::forCurrentThread();
            auto deltaTime = endTime - startTime;
            
            JSStringRelease(script);

            if ((deltaTime >= timeBeforeExtendedDeadline) && (deltaTime < timeAfterExtendedDeadline) && (extendTerminateCallbackCalled == 2) && exception)
                printf("PASS: %s script timeout was extended as expected.\n", tierName);
            else {
                if (deltaTime < timeBeforeExtendedDeadline)
                    printf("FAIL: %s script timeout was not extended as expected.\n", tierName);
                else if (deltaTime >= timeAfterExtendedDeadline)
                    printf("FAIL: %s script did not timeout.\n", tierName);
                
                if (extendTerminateCallbackCalled < 1)
                    printf("FAIL: %s script timeout callback was not called.\n", tierName);
                if (extendTerminateCallbackCalled < 2)
                    printf("FAIL: %s script timeout callback was not called after timeout extension.\n", tierName);
                
                if (!exception)
                    printf("FAIL: %s TerminationException was not thrown during timeout extension test.\n", tierName);
                
                failed = true;
            }
        }

#if HAVE(MACH_EXCEPTIONS)
        /* Test script timeout from dispatch queue: */
        timeLimit = 100_ms + tierAdjustment;
        JSContextGroupSetExecutionTimeLimit(contextGroup, timeLimit.seconds(), dispatchTermitateCallback, nullptr);
        {
            Seconds timeAfterWatchdogShouldHaveFired = 300_ms + tierAdjustment;

            CString scriptText = makeString(
                "function foo() {"
                    "var startTime = currentCPUTime();"
                    "while (true) {"
                        "for (var i = 0; i < 1000; i++);"
                            "if (currentCPUTime() - startTime > "_s, timeAfterWatchdogShouldHaveFired.seconds(), ") break;"
                    "}"
                "}"
                "foo();"_s
            ).utf8();

            JSStringRef script = JSStringCreateWithUTF8CString(scriptText.data());
            exception = nullptr;
            dispatchTerminateCallbackCalled = false;

            // We have to do this since blocks can only capture things as const.
            JSGlobalContextRef& contextRef = context;
            JSStringRef& scriptRef = script;
            JSValueRef& exceptionRef = exception;

            Lock syncLock;
            Lock& syncLockRef = syncLock;
            Condition synchronize;
            Condition& synchronizeRef = synchronize;
            bool didSynchronize = false;
            bool& didSynchronizeRef = didSynchronize;

            Seconds startTime;
            Seconds endTime;

            Seconds& startTimeRef = startTime;
            Seconds& endTimeRef = endTime;

            dispatch_group_t group = dispatch_group_create();
            dispatch_group_async(group, dispatch_get_global_queue(0, 0), ^{
                startTimeRef = CPUTime::forCurrentThread();
                JSEvaluateScript(contextRef, scriptRef, nullptr, nullptr, 1, &exceptionRef);
                endTimeRef = CPUTime::forCurrentThread();
                Locker locker { syncLockRef };
                didSynchronizeRef = true;
                synchronizeRef.notifyAll();
            });

            Locker locker { syncLock };
            synchronize.wait(syncLock, [&] { return didSynchronize; });

            if (((endTime - startTime) < timeAfterWatchdogShouldHaveFired) && dispatchTerminateCallbackCalled)
                printf("PASS: %s script on dispatch queue timed out as expected.\n", tierName);
            else {
                if ((endTime - startTime) >= timeAfterWatchdogShouldHaveFired)
                    printf("FAIL: %s script on dispatch queue did not time out as expected.\n", tierName);
                if (!shouldTerminateCallbackWasCalled)
                    printf("FAIL: %s script on dispatch queue timeout callback was not called.\n", tierName);
                failed = true;
            }

            JSStringRelease(script);
        }
#endif

        JSGlobalContextRelease(context);

        Options::setOptions(savedOptionsBuilder.toString().ascii().data());
    }
    
    return failed;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
