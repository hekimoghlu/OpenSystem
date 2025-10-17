/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#include "JSObjectGetProxyTargetTest.h"

#include "APICast.h"
#include "IntegrityInlines.h"
#include "JSCInlines.h"
#include "JSGlobalProxyInlines.h"
#include "JSObjectRefPrivate.h"
#include "JavaScript.h"
#include "ProxyObject.h"
#include <wtf/text/ASCIILiteral.h>

using namespace JSC;

int testJSObjectGetProxyTarget()
{
    bool overallResult = true;
    
    printf("JSObjectGetProxyTargetTest:\n");
    
    auto test = [&] (ASCIILiteral description, bool currentResult) {
        SAFE_PRINTF("    %s: %s\n", description, currentResult ? "PASS"_s : "FAIL"_s);
        overallResult &= currentResult;
    };
    
    JSContextGroupRef group = JSContextGroupCreate();
    JSGlobalContextRef context = JSGlobalContextCreateInGroup(group, nullptr);
    
    VM& vm = *toJS(group);
    JSObjectRef globalObjectProxy = JSContextGetGlobalObject(context);

    JSGlobalObject* globalObjectObject;
    JSObjectRef globalObjectRef;
    JSGlobalProxy* jsProxyObject;

    {
        JSLockHolder locker(vm);
        JSGlobalProxy* globalObjectProxyObject = jsCast<JSGlobalProxy*>(toJS(globalObjectProxy));
        globalObjectObject = jsCast<JSGlobalObject*>(globalObjectProxyObject->target());
        Structure* proxyStructure = JSGlobalProxy::createStructure(vm, globalObjectObject, globalObjectObject->objectPrototype());
        globalObjectRef = toRef(jsCast<JSObject*>(globalObjectObject));
        jsProxyObject = JSGlobalProxy::create(vm, proxyStructure);
    }
    
    JSObjectRef array = JSObjectMakeArray(context, 0, nullptr, nullptr);

    ProxyObject* proxyObjectObject;

    {
        JSLockHolder locker(vm);
        Structure* emptyObjectStructure = JSFinalObject::createStructure(vm, globalObjectObject, globalObjectObject->objectPrototype(), 0);
        JSObject* handler = JSFinalObject::create(vm, emptyObjectStructure);
        proxyObjectObject = ProxyObject::create(globalObjectObject, toJS(array), handler);
    }

    JSObjectRef jsProxy = toRef(jsProxyObject);
    JSObjectRef proxyObject = toRef(proxyObjectObject);
    
    test("proxy target of null is null", !JSObjectGetProxyTarget(nullptr));
    test("proxy target of non-proxy is null", !JSObjectGetProxyTarget(array));
    test("proxy target of uninitialized JSGlobalProxy is null", !JSObjectGetProxyTarget(jsProxy));
    
    {
        JSLockHolder locker(vm);
        jsProxyObject->setTarget(vm, globalObjectObject);
    }
    
    test("proxy target of initialized JSGlobalProxy works", JSObjectGetProxyTarget(jsProxy) == globalObjectRef);
    
    test("proxy target of ProxyObject works", JSObjectGetProxyTarget(proxyObject) == array);
    
    test("proxy target of GlobalObject is the globalObject", JSObjectGetProxyTarget(globalObjectProxy) == globalObjectRef);

    JSGlobalContextRelease(context);
    JSContextGroupRelease(group);

    SAFE_PRINTF("JSObjectGetProxyTargetTest: %s\n", overallResult ? "PASS"_s : "FAIL"_s);
    return !overallResult;
}

