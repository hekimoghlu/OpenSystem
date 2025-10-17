/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#include "MultithreadedMultiVMExecutionTest.h"

#include "InitializeThreading.h"
#include "JavaScript.h"
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <wtf/MainThread.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>

static int failuresFound = 0;

static std::vector<std::thread>& threadsList()
{
    static std::vector<std::thread>* list;
    static std::once_flag flag;
    std::call_once(flag, [] () {
        list = new std::vector<std::thread>();
    });
    return *list;
}

void startMultithreadedMultiVMExecutionTest()
{
    WTF::initializeMainThread();
    JSC::initialize();

#define CHECK(condition, threadNumber, count, message) do { \
        if (!condition) { \
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN \
            printf("FAIL: MultithreadedMultiVMExecutionTest: %d %d %s\n", threadNumber, count, message); \
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END \
            failuresFound++; \
        } \
    } while (false)

    auto task = [](int threadNumber) {
        int ret = 0;
        std::string scriptString =
            "const AAA = {A:0, B:1, C:2, D:3};"
            "class Preconditions { static checkArgument(e,t) { if (!e) throw t }};"
            "1 + 2";

        for (int i = 0; i < 1000; ++i) {
            JSClassRef jsClass = JSClassCreate(&kJSClassDefinitionEmpty);
            CHECK(jsClass, threadNumber, i, "global object class creation");
            JSContextGroupRef contextGroup = JSContextGroupCreate();
            CHECK(contextGroup, threadNumber, i, "group creation");
            JSGlobalContextRef context = JSGlobalContextCreateInGroup(contextGroup, jsClass);
            CHECK(context, threadNumber, i, "ctx creation");

            JSStringRef jsScriptString = JSStringCreateWithUTF8CString(scriptString.c_str());
            CHECK(jsScriptString, threadNumber, i, "script to jsString");

            JSValueRef exception = nullptr;
            JSValueRef jsScript = JSEvaluateScript(context, jsScriptString, nullptr, nullptr, 0, &exception);
            CHECK(!exception, threadNumber, i, "script eval no exception");
            if (exception) {
                JSStringRef string = JSValueToStringCopy(context, exception, nullptr);
                if (string) {
                    Vector<char> buffer(JSStringGetMaximumUTF8CStringSize(string));
                    JSStringGetUTF8CString(string, buffer.data(), buffer.size());
                    SAFE_PRINTF("FAIL: MultithreadedMultiVMExecutionTest: %d %d %s\n", threadNumber, i, CString(buffer.span()));
                    JSStringRelease(string);
                } else
                    printf("FAIL: MultithreadedMultiVMExecutionTest: %d %d stringifying exception failed\n", threadNumber, i);
            }
            CHECK(jsScript, threadNumber, i, "script eval");
            JSStringRelease(jsScriptString);

            JSGlobalContextRelease(context);
            JSContextGroupRelease(contextGroup);
            JSClassRelease(jsClass);
        }

        return ret;
    };
    for (int t = 0; t < 8; ++t)
        threadsList().push_back(std::thread(task, t));
}

int finalizeMultithreadedMultiVMExecutionTest()
{
    auto& threads = threadsList();
    for (auto& thread : threads)
        thread.join();

    SAFE_PRINTF("%s: MultithreadedMultiVMExecutionTest\n", failuresFound ? "FAIL"_s : "PASS"_s);
    return (failuresFound > 0);
}
