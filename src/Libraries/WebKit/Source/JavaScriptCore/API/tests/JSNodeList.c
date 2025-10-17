/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "JSValueRef.h"
#include <wtf/Assertions.h>

static JSValueRef JSNodeList_item(JSContextRef context, JSObjectRef object, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    UNUSED_PARAM(object);

    if (argumentCount > 0) {
        NodeList* nodeList = JSObjectGetPrivate(thisObject);
        ASSERT(nodeList);
        Node* node = NodeList_item(nodeList, (unsigned)JSValueToNumber(context, arguments[0], exception));
        if (node)
            return JSNode_new(context, node);
    }
    
    return JSValueMakeUndefined(context);
}

static const JSStaticFunction JSNodeList_staticFunctions[] = {
    { "item", JSNodeList_item, kJSPropertyAttributeDontDelete },
    { 0, 0, 0 }
};

static JSValueRef JSNodeList_length(JSContextRef context, JSObjectRef thisObject, JSStringRef propertyName, JSValueRef* exception)
{
    UNUSED_PARAM(propertyName);
    UNUSED_PARAM(exception);
    
    NodeList* nodeList = JSObjectGetPrivate(thisObject);
    ASSERT(nodeList);
    return JSValueMakeNumber(context, NodeList_length(nodeList));
}

static const JSStaticValue JSNodeList_staticValues[] = {
    { "length", JSNodeList_length, NULL, kJSPropertyAttributeReadOnly | kJSPropertyAttributeDontDelete },
    { 0, 0, 0, 0 }
};

static JSValueRef JSNodeList_getProperty(JSContextRef context, JSObjectRef thisObject, JSStringRef propertyName, JSValueRef* exception)
{
    NodeList* nodeList = JSObjectGetPrivate(thisObject);
    ASSERT(nodeList);
    double index = JSValueToNumber(context, JSValueMakeString(context, propertyName), exception);
    unsigned uindex = (unsigned)index;
    if (uindex == index) { /* false for NaN */
        Node* node = NodeList_item(nodeList, uindex);
        if (node)
            return JSNode_new(context, node);
    }
    
    return NULL;
}

static void JSNodeList_initialize(JSContextRef context, JSObjectRef thisObject)
{
    UNUSED_PARAM(context);

    NodeList* nodeList = JSObjectGetPrivate(thisObject);
    ASSERT(nodeList);
    
    NodeList_ref(nodeList);
}

static void JSNodeList_finalize(JSObjectRef thisObject)
{
    NodeList* nodeList = JSObjectGetPrivate(thisObject);
    ASSERT(nodeList);

    NodeList_deref(nodeList);
}

static JSClassRef JSNodeList_class(JSContextRef context)
{
    UNUSED_PARAM(context);

    static JSClassRef jsClass;
    if (!jsClass) {
        JSClassDefinition definition = kJSClassDefinitionEmpty;
        definition.staticValues = JSNodeList_staticValues;
        definition.staticFunctions = JSNodeList_staticFunctions;
        definition.getProperty = JSNodeList_getProperty;
        definition.initialize = JSNodeList_initialize;
        definition.finalize = JSNodeList_finalize;

        jsClass = JSClassCreate(&definition);
    }
    
    return jsClass;
}

JSObjectRef JSNodeList_new(JSContextRef context, NodeList* nodeList)
{
    return JSObjectMake(context, JSNodeList_class(context), nodeList);
}
