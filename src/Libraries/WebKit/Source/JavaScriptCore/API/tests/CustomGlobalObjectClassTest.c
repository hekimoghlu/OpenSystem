/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include "CustomGlobalObjectClassTest.h"

#include <JavaScriptCore/JSObjectRefPrivate.h>
#include <JavaScriptCore/JavaScript.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

extern bool assertTrue(bool value, const char* message);

static bool executedCallback = false;

static JSValueRef jsDoSomething(JSContextRef ctx, JSObjectRef function, JSObjectRef thisObject, size_t argc, const JSValueRef args[], JSValueRef* exception)
{
    (void)function;
    (void)thisObject;
    (void)argc;
    (void)args;
    (void)exception;
    executedCallback = true;
    return JSValueMakeNull(ctx);
}

static const JSStaticFunction bridgedFunctions[] = {
    {"doSomething", jsDoSomething, kJSPropertyAttributeDontDelete},
    {0, 0, 0},
};

static JSClassRef bridgedObjectClass = NULL;
static JSClassDefinition bridgedClassDef;

static JSClassRef jsClassRef(void)
{
    if (!bridgedObjectClass) {
        bridgedClassDef = kJSClassDefinitionEmpty;
        bridgedClassDef.className = "BridgedObject";
        bridgedClassDef.staticFunctions = bridgedFunctions;
        bridgedObjectClass = JSClassCreate(&bridgedClassDef);
    }
    return bridgedObjectClass;
}

void customGlobalObjectClassTest(void)
{
    JSClassRef bridgedObjectJsClassRef = jsClassRef();
    JSGlobalContextRef globalContext = JSGlobalContextCreate(bridgedObjectJsClassRef);
    
    JSObjectRef globalObj = JSContextGetGlobalObject(globalContext);
    
    JSPropertyNameArrayRef propertyNames = JSObjectCopyPropertyNames(globalContext, globalObj);
    size_t propertyCount = JSPropertyNameArrayGetCount(propertyNames);
    assertTrue(propertyCount == 1, "Property count == 1");
    
    JSStringRef propertyNameRef = JSPropertyNameArrayGetNameAtIndex(propertyNames, 0);
    size_t propertyNameLength = JSStringGetLength(propertyNameRef);
    size_t bufferSize = sizeof(char) * (propertyNameLength + 1);
    char* buffer = (char*)malloc(bufferSize);
    JSStringGetUTF8CString(propertyNameRef, buffer, bufferSize);
    buffer[propertyNameLength] = '\0';
    assertTrue(!strncmp(buffer, "doSomething", propertyNameLength), "First property name is doSomething");
    free(buffer);
    
    bool hasMethod = JSObjectHasProperty(globalContext, globalObj, propertyNameRef);
    assertTrue(hasMethod, "Property found by name");
    
    JSValueRef doSomethingProperty =
    JSObjectGetProperty(globalContext, globalObj, propertyNameRef, NULL);
    assertTrue(!JSValueIsUndefined(globalContext, doSomethingProperty), "Property is defined");
    
    bool globalObjectClassMatchesClassRef = JSValueIsObjectOfClass(globalContext, globalObj, bridgedObjectJsClassRef);
    assertTrue(globalObjectClassMatchesClassRef, "Global object is the right class");
    
    JSStringRef script = JSStringCreateWithUTF8CString("doSomething();");
    JSEvaluateScript(globalContext, script, NULL, NULL, 1, NULL);
    JSStringRelease(script);

    assertTrue(executedCallback, "Executed custom global object callback");

    JSGlobalContextRelease(globalContext);
}

void globalObjectSetPrototypeTest(void)
{
    JSClassDefinition definition = kJSClassDefinitionEmpty;
    definition.className = "Global";
    JSClassRef global = JSClassCreate(&definition);
    JSGlobalContextRef context = JSGlobalContextCreate(global);
    JSObjectRef object = JSContextGetGlobalObject(context);

    JSValueRef originalPrototype = JSObjectGetPrototype(context, object);
    JSObjectRef above = JSObjectMake(context, 0, 0);
    JSObjectSetPrototype(context, object, above);
    JSValueRef prototypeAfterChangingAttempt = JSObjectGetPrototype(context, object);
    assertTrue(JSValueIsStrictEqual(context, prototypeAfterChangingAttempt, originalPrototype), "Global object's [[Prototype]] cannot be changed after instantiating it");
    JSGlobalContextRelease(context);
}

void globalObjectPrivatePropertyTest(void)
{
    JSClassDefinition definition = kJSClassDefinitionEmpty;
    definition.className = "Global";
    JSClassRef global = JSClassCreate(&definition);
    JSGlobalContextRef context = JSGlobalContextCreate(global);
    JSObjectRef globalObject = JSContextGetGlobalObject(context);

    JSStringRef privateName = JSStringCreateWithUTF8CString("private");
    JSValueRef privateValue = JSValueMakeString(context, privateName);
    assertTrue(JSObjectSetPrivateProperty(context, globalObject, privateName, privateValue), "JSObjectSetPrivateProperty succeeded");
    JSValueRef result = JSObjectGetPrivateProperty(context, globalObject, privateName);
    assertTrue(JSValueIsStrictEqual(context, privateValue, result), "privateValue === \"private\"");

    assertTrue(JSObjectDeletePrivateProperty(context, globalObject, privateName), "JSObjectDeletePrivateProperty succeeded");
    result = JSObjectGetPrivateProperty(context, globalObject, privateName);
    assertTrue(JSValueIsNull(context, result), "Deleted private property is indeed no longer present");

    JSStringRelease(privateName);
    JSGlobalContextRelease(context);
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
