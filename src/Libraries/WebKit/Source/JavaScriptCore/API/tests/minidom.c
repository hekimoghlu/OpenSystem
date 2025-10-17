/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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

#include "JSContextRef.h"
#include "JSNode.h"
#include "JSObjectRef.h"
#include "JSStringRef.h"
#include <stdio.h>
#include <stdlib.h>
#include <wtf/Assertions.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

static char* createStringWithContentsOfFile(const char* fileName);
static JSValueRef print(JSContextRef context, JSObjectRef object, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception);

int main(int argc, char* argv[])
{
    const char *scriptPath = "minidom.js";
    if (argc > 1) {
        scriptPath = argv[1];
    }
    
    JSGlobalContextRef context = JSGlobalContextCreateInGroup(NULL, NULL);
    JSObjectRef globalObject = JSContextGetGlobalObject(context);
    
    JSStringRef printIString = JSStringCreateWithUTF8CString("print");
    JSObjectSetProperty(context, globalObject, printIString, JSObjectMakeFunctionWithCallback(context, printIString, print), kJSPropertyAttributeNone, NULL);
    JSStringRelease(printIString);
    
    JSStringRef node = JSStringCreateWithUTF8CString("Node");
    JSObjectSetProperty(context, globalObject, node, JSObjectMakeConstructor(context, JSNode_class(context), JSNode_construct), kJSPropertyAttributeNone, NULL);
    JSStringRelease(node);
    
    char* scriptUTF8 = createStringWithContentsOfFile(scriptPath);
    JSStringRef script = JSStringCreateWithUTF8CString(scriptUTF8);
    JSValueRef exception;
    JSValueRef result = JSEvaluateScript(context, script, NULL, NULL, 1, &exception);
    if (result)
        printf("PASS: Test script executed successfully.\n");
    else {
        printf("FAIL: Test script threw exception:\n");
        JSStringRef exceptionIString = JSValueToStringCopy(context, exception, NULL);
        size_t exceptionUTF8Size = JSStringGetMaximumUTF8CStringSize(exceptionIString);
        char* exceptionUTF8 = (char*)malloc(exceptionUTF8Size);
        JSStringGetUTF8CString(exceptionIString, exceptionUTF8, exceptionUTF8Size);
        printf("%s\n", exceptionUTF8);
        free(exceptionUTF8);
        JSStringRelease(exceptionIString);
    }
    JSStringRelease(script);
    free(scriptUTF8);

    globalObject = 0;
    JSGlobalContextRelease(context);
    printf("PASS: Program exited normally.\n");
    return 0;
}

static JSValueRef print(JSContextRef context, JSObjectRef object, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(object);
    UNUSED_PARAM(thisObject);

    if (argumentCount > 0) {
        JSStringRef string = JSValueToStringCopy(context, arguments[0], exception);
        size_t numChars = JSStringGetMaximumUTF8CStringSize(string);
        char* stringUTF8 = (char*)malloc(numChars);
        JSStringGetUTF8CString(string, stringUTF8, numChars);
        printf("%s\n", stringUTF8);
        JSStringRelease(string);
        free(stringUTF8);
    }
    
    return JSValueMakeUndefined(context);
}

static char* createStringWithContentsOfFile(const char* fileName)
{
    char* buffer;
    
    size_t buffer_size = 0;
    size_t buffer_capacity = 1024;
    buffer = (char*)malloc(buffer_capacity);
    
    FILE* f = fopen(fileName, "r");
    if (!f) {
        fprintf(stderr, "Could not open file: %s\n", fileName);
        free(buffer);
        return 0;
    }
    
    while (!feof(f) && !ferror(f)) {
        buffer_size += fread(buffer + buffer_size, 1, buffer_capacity - buffer_size, f);
        if (buffer_size == buffer_capacity) { /* guarantees space for trailing '\0' */
            buffer_capacity *= 2;
            buffer = (char*)realloc(buffer, buffer_capacity);
            ASSERT(buffer);
        }
        
        ASSERT(buffer_size < buffer_capacity);
    }
    fclose(f);
    buffer[buffer_size] = '\0';
    
    return buffer;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
