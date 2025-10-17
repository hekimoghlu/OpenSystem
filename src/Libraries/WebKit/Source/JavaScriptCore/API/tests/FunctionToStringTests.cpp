/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include "FunctionToStringTests.h"

#include "InitializeThreading.h"
#include "JavaScript.h"
#include <stdio.h>
#include <wtf/text/ASCIILiteral.h>

int testFunctionToString()
{
    const auto inputScript = R"(
        var valid = true;

        function foo2   () {}
        valid &&= foo2.toString() == "function foo2   () {}"

        function       foo3()   {}
        valid &&= foo3.toString() == "function       foo3()   {}"

        function*  fooGen   (){}
        valid &&= fooGen.toString() == "function*  fooGen   (){}"

        async function fnasync() {}
        valid &&= fnasync.toString() == "async function fnasync() {}"

        let f1 = async function() {}
        valid &&= f1.toString() == "async function() {}"

        let f2 = async()=>{}
        valid &&= f2.toString() == "async()=>{}"

        let f3 = async  ()    =>     {}
        valid &&= f3.toString() == "async  ()    =>     {}"

        let asyncGenExpr = async function*()  {}
        valid &&= asyncGenExpr.toString() == "async function*()  {}"

        async function* asyncGenDecl() {}
        valid &&= asyncGenDecl.toString() == "async function* asyncGenDecl() {}"

        async  function  *  fga  (  x  ,  y  )  {  ;  ;  }
        valid &&= fga.toString() == "async  function  *  fga  (  x  ,  y  )  {  ;  ;  }"

        let fDeclAndExpr = { async f  (  )  {  } }.f;
        valid &&= fDeclAndExpr.toString() == "async f  (  )  {  }"

        let fAsyncGenInStaticMethod = class { static async  *  f  (  )  {  } }.f
        valid &&= fAsyncGenInStaticMethod.toString() == "async  *  f  (  )  {  }"

        let fPropFuncGen = { *  f  (  )  {  } }.f;
        valid &&= fPropFuncGen.toString() == "*  f  (  )  {  }"

        let fGetter = Object.getOwnPropertyDescriptor(class { static get  f  (  )  {  } }, "f").get
        valid &&= fGetter.toString() == "get  f  (  )  {  }"

        let g = class { static [  "g"  ]  (  )  {  } }.g
        valid &&= g.toString() == '[  "g"  ]  (  )  {  }'

        let fMethodObject = { f  (  )  {  } }.f
        valid &&= fMethodObject.toString() == "f  (  )  {  }"

        let fComputedProp = { [  "f"  ]  (  )  {  } }.f
        valid &&= fComputedProp.toString() == '[  "f"  ]  (  )  {  }'
    
        let gAsyncPropFunc = { async  [  "g"  ]  (  )  {  } }.g
        valid &&= gAsyncPropFunc.toString() == 'async  [  "g"  ]  (  )  {  }'
    
        valid
    )";

    JSC::initialize();

    JSGlobalContextRef context = JSGlobalContextCreateInGroup(nullptr, nullptr);
    JSStringRef script = JSStringCreateWithUTF8CString(inputScript);
    JSValueRef exception = nullptr;
    JSValueRef resultRef = JSEvaluateScript(context, script, nullptr, nullptr, 1, &exception);
    JSStringRelease(script);

    auto failed = false;
    if (!JSValueIsBoolean(context, resultRef) || !JSValueToBoolean(context, resultRef))
        failed = true;

    JSGlobalContextRelease(context);
    SAFE_PRINTF("%s: function toString tests.\n", failed ? "FAIL"_s : "PASS"_s);

    return failed;
}
