/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#import "WebJSPDFDoc.h"

#import "WebDataSource.h"
#import "WebDelegateImplementationCaching.h"
#import "WebFrame.h"
#import "WebUIDelegate.h"
#import "WebView.h"
#import <JavaScriptCore/JSObjectRef.h>

static void jsPDFDocInitialize(JSContextRef ctx, JSObjectRef object)
{
    CFRetain(JSObjectGetPrivate(object));
}

static void jsPDFDocFinalize(JSObjectRef object)
{
    CFRelease(JSObjectGetPrivate(object));
}

static JSClassRef jsPDFDocClass(void);

static JSValueRef jsPDFDocPrint(JSContextRef ctx, JSObjectRef function, JSObjectRef thisObject, size_t argumentCount, const JSValueRef arguments[], JSValueRef* exception)
{
    if (!JSValueIsObjectOfClass(ctx, thisObject, jsPDFDocClass()))
        return JSValueMakeUndefined(ctx);

    WebDataSource *dataSource = (__bridge WebDataSource *)JSObjectGetPrivate(thisObject);

    WebView *webView = [[dataSource webFrame] webView];
    CallUIDelegate(webView, @selector(webView:printFrameView:), [[dataSource webFrame] frameView]);

    return JSValueMakeUndefined(ctx);
}

static JSClassRef jsPDFDocClass(void)
{
    static JSStaticFunction jsPDFDocStaticFunctions[] = {
        { "print", jsPDFDocPrint, kJSPropertyAttributeReadOnly | kJSPropertyAttributeDontDelete },
        { 0, 0, 0 },
    };

    static JSClassDefinition jsPDFDocClassDefinition = {
        0,
        kJSClassAttributeNone,
        "Doc",
        0,
        0,
        jsPDFDocStaticFunctions,
        jsPDFDocInitialize, jsPDFDocFinalize, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    static JSClassRef jsPDFDocClass = JSClassCreate(&jsPDFDocClassDefinition);
    return jsPDFDocClass;
}

JSObjectRef makeJSPDFDoc(JSContextRef ctx, WebDataSource *dataSource)
{
    return JSObjectMake(ctx, jsPDFDocClass(), (__bridge void*)dataSource);
}
