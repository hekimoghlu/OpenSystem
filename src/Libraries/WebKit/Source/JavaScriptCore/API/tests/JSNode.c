/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

#include "JSNode.h"
#include "JSNodeList.h"
#include "JSObjectRef.h"
#include "JSStringRef.h"
#include "JSValueRef.h"
#include "Node.h"
#include "NodeList.h"
#include <wtf/Assertions.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

static JSValueRef JSNode_appendChild(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(function);

    /* Example of throwing a type error for invalid values */
    if (!JSValueIsObjectOfClass(context, thisObject, JSNode_class(context))) {
        JSStringRef message = JSStringCreateWithUTF8CString("TypeError: appendChild can only be called on nodes");
        *exception = JSValueMakeString(context, message);
        JSStringRelease(message);
    } else if (argumentCount < 1 || !JSValueIsObjectOfClass(context, arguments[0], JSNode_class(context))) {
        JSStringRef message = JSStringCreateWithUTF8CString("TypeError: first argument to appendChild must be a node");
        *exception = JSValueMakeString(context, message);
        JSStringRelease(message);
    } else {
        Node* node = JSObjectGetPrivate(thisObject);
        Node* child = JSObjectGetPrivate(JSValueToObject(context, arguments[0], NULL));

        Node_appendChild(node, child);
    }

    return JSValueMakeUndefined(context);
}

static JSValueRef JSNode_removeChild(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(function);

    /* Example of ignoring invalid values */
    if (argumentCount > 0) {
        if (JSValueIsObjectOfClass(context, thisObject, JSNode_class(context))) {
            if (JSValueIsObjectOfClass(context, arguments[0], JSNode_class(context))) {
                Node* node = JSObjectGetPrivate(thisObject);
                Node* child = JSObjectGetPrivate(JSValueToObject(context, arguments[0], exception));
                
                Node_removeChild(node, child);
            }
        }
    }
    
    return JSValueMakeUndefined(context);
}

static JSValueRef JSNode_replaceChild(JSContextRef context, JSObjectRef function, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(function);
    
    if (argumentCount > 1) {
        if (JSValueIsObjectOfClass(context, thisObject, JSNode_class(context))) {
            if (JSValueIsObjectOfClass(context, arguments[0], JSNode_class(context))) {
                if (JSValueIsObjectOfClass(context, arguments[1], JSNode_class(context))) {
                    Node* node = JSObjectGetPrivate(thisObject);
                    Node* newChild = JSObjectGetPrivate(JSValueToObject(context, arguments[0], exception));
                    Node* oldChild = JSObjectGetPrivate(JSValueToObject(context, arguments[1], exception));
                    
                    Node_replaceChild(node, newChild, oldChild);
                }
            }
        }
    }
    
    return JSValueMakeUndefined(context);
}

static const JSStaticFunction JSNode_staticFunctions[] = {
    { "appendChild", JSNode_appendChild, kJSPropertyAttributeDontDelete },
    { "removeChild", JSNode_removeChild, kJSPropertyAttributeDontDelete },
    { "replaceChild", JSNode_replaceChild, kJSPropertyAttributeDontDelete },
    { 0, 0, 0 }
};

static JSValueRef JSNode_getNodeType(JSContextRef context, JSObjectRef object, JSStringRef propertyName, JSValueRef* exception)
{
    UNUSED_PARAM(propertyName);
    UNUSED_PARAM(exception);

    Node* node = JSObjectGetPrivate(object);
    if (node) {
        JSStringRef nodeType = JSStringCreateWithUTF8CString(node->nodeType);
        JSValueRef value = JSValueMakeString(context, nodeType);
        JSStringRelease(nodeType);
        return value;
    }
    
    return NULL;
}

static JSValueRef JSNode_getChildNodes(JSContextRef context, JSObjectRef thisObject, JSStringRef propertyName, JSValueRef* exception)
{
    UNUSED_PARAM(propertyName);
    UNUSED_PARAM(exception);

    Node* node = JSObjectGetPrivate(thisObject);
    ASSERT(node);
    return JSNodeList_new(context, NodeList_new(node));
}

static JSValueRef JSNode_getFirstChild(JSContextRef context, JSObjectRef object, JSStringRef propertyName, JSValueRef* exception)
{
    UNUSED_PARAM(object);
    UNUSED_PARAM(propertyName);
    UNUSED_PARAM(exception);
    
    return JSValueMakeUndefined(context);
}

static const JSStaticValue JSNode_staticValues[] = {
    { "nodeType", JSNode_getNodeType, NULL, kJSPropertyAttributeDontDelete | kJSPropertyAttributeReadOnly },
    { "childNodes", JSNode_getChildNodes, NULL, kJSPropertyAttributeDontDelete | kJSPropertyAttributeReadOnly },
    { "firstChild", JSNode_getFirstChild, NULL, kJSPropertyAttributeDontDelete | kJSPropertyAttributeReadOnly },
    { 0, 0, 0, 0 }
};

static void JSNode_initialize(JSContextRef context, JSObjectRef object)
{
    UNUSED_PARAM(context);

    Node* node = JSObjectGetPrivate(object);
    ASSERT(node);

    Node_ref(node);
}

static void JSNode_finalize(JSObjectRef object)
{
    Node* node = JSObjectGetPrivate(object);
    ASSERT(node);

    Node_deref(node);
}

JSClassRef JSNode_class(JSContextRef context)
{
    UNUSED_PARAM(context);

    static JSClassRef jsClass;
    if (!jsClass) {
        JSClassDefinition definition = kJSClassDefinitionEmpty;
        definition.staticValues = JSNode_staticValues;
        definition.staticFunctions = JSNode_staticFunctions;
        definition.initialize = JSNode_initialize;
        definition.finalize = JSNode_finalize;

        jsClass = JSClassCreate(&definition);
    }
    return jsClass;
}

JSObjectRef JSNode_new(JSContextRef context, Node* node)
{
    return JSObjectMake(context, JSNode_class(context), node);
}

JSObjectRef JSNode_construct(JSContextRef context, JSObjectRef object, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(object);
    UNUSED_PARAM(argumentCount);
    UNUSED_PARAM(arguments);
    UNUSED_PARAM(exception);

    return JSNode_new(context, Node_new());
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
