/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
#include "FunctionOverridesTest.h"

#include "FunctionOverrides.h"
#include "InitializeThreading.h"
#include "JavaScript.h"
#include "Options.h"

using JSC::Options;

int testFunctionOverrides()
{
    bool failed = false;

    JSC::initialize();

    const char* oldFunctionOverrides = Options::functionOverrides();
    
    Options::functionOverrides() = "./testapiScripts/testapi-function-overrides.js";
    JSC::FunctionOverrides::reinstallOverrides();

    JSGlobalContextRef context = JSGlobalContextCreateInGroup(nullptr, nullptr);

    JSObjectRef globalObject = JSContextGetGlobalObject(context);
    ASSERT_UNUSED(globalObject, JSValueIsObject(context, globalObject));

    const char* scriptString =
        "var str = '';" "\n"
        "function f1() { /* Original f1 */ }" "\n"
        "str += f1 + '\\n';" "\n"
        "var f2 = function() {" "\n"
        "    // Original f2" "\n"
        "}" "\n"
        "str += f2 + '\\n';" "\n"
        "str += (function() { /* Original f3 */ }) + '\\n';" "\n"
        "var f4Source = '/* Original f4 */'" "\n"
        "var f4 =  new Function(f4Source);" "\n"
        "str += f4 + '\\n';" "\n"
        "\n"
        "var expectedStr =" "\n"
        "'function f1() { /* Overridden f1 */ }\\n"
        "function() { /* Overridden f2 */ }\\n"
        "function() { /* Overridden f3 */ }\\n"
        "function anonymous(\\n) { /* Overridden f4 */ }\\n';"
        "var result = (str == expectedStr);" "\n"
        "result";

    JSStringRef script = JSStringCreateWithUTF8CString(scriptString);
    JSValueRef exception = nullptr;
    JSValueRef resultRef = JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
    JSStringRelease(script);

    if (!JSValueIsBoolean(context, resultRef) || !JSValueToBoolean(context, resultRef))
        failed = true;

    JSGlobalContextRelease(context);
    
    JSC::Options::functionOverrides() = oldFunctionOverrides;
    JSC::FunctionOverrides::reinstallOverrides();

    SAFE_PRINTF("%s: function override tests.\n", failed ? "FAIL"_s : "PASS"_s);

    return failed;
}
