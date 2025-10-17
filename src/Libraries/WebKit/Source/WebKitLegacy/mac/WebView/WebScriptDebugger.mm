/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#import "WebScriptDebugger.h"

#import "WebDelegateImplementationCaching.h"
#import "WebFrameInternal.h"
#import "WebScriptDebugDelegate.h"
#import "WebViewInternal.h"
#import <JavaScriptCore/Breakpoint.h>
#import <JavaScriptCore/DebuggerCallFrame.h>
#import <JavaScriptCore/JSGlobalObject.h>
#import <JavaScriptCore/SourceProvider.h>
#import <JavaScriptCore/StrongInlines.h>
#import <WebCore/JSDOMWindow.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/ScriptController.h>
#import <wtf/URL.h>

@interface WebScriptCallFrame (WebScriptDebugDelegateInternalForDebugger)
- (WebScriptCallFrame *)_initWithGlobalObject:(WebScriptObject *)globalObj functionName:(String)functionName exceptionValue:(JSC::JSValue)exceptionValue;
@end

static NSString *toNSString(JSC::SourceProvider* sourceProvider)
{
    const String& sourceString = sourceProvider->source().toString();
    if (sourceString.isEmpty())
        return nil;
    return sourceString;
}

static WebFrame *toWebFrame(JSC::JSGlobalObject* globalObject)
{
    auto* window = static_cast<WebCore::JSDOMWindow*>(globalObject);
    return kit(dynamicDowncast<WebCore::LocalFrame>(window->wrapped().frame()));
}

WebScriptDebugger::WebScriptDebugger(JSC::JSGlobalObject* globalObject)
    : Debugger(globalObject->vm())
    , m_callingDelegate(false)
    , m_globalObject(globalObject->vm(), globalObject)
{
    setPauseOnAllExceptionsBreakpoint(JSC::Breakpoint::create(JSC::noBreakpointID));
    deactivateBreakpoints();
    attach(globalObject);
}

// callbacks - relay to delegate
void WebScriptDebugger::sourceParsed(JSC::JSGlobalObject* lexicalGlobalObject, JSC::SourceProvider* sourceProvider, int errorLine, const String& errorMsg)
{
    if (m_callingDelegate)
        return;

    m_callingDelegate = true;

    NSString *nsSource = toNSString(sourceProvider);
    NSURL *nsURL = sourceProvider->sourceOrigin().url();
    int firstLine = sourceProvider->startPosition().m_line.oneBasedInt();

    JSC::VM& vm = lexicalGlobalObject->vm();
    WebFrame *webFrame = toWebFrame(vm.deprecatedVMEntryGlobalObject(lexicalGlobalObject));
    WebView *webView = [webFrame webView];
    WebScriptDebugDelegateImplementationCache* implementations = WebViewGetScriptDebugDelegateImplementations(webView);

    if (errorLine == -1) {
        if (implementations->didParseSourceFunc) {
            if (implementations->didParseSourceExpectsBaseLineNumber)
                CallScriptDebugDelegate(implementations->didParseSourceFunc, webView, @selector(webView:didParseSource:baseLineNumber:fromURL:sourceId:forWebFrame:), nsSource, firstLine, nsURL, sourceProvider->asID(), webFrame);
            else
                CallScriptDebugDelegate(implementations->didParseSourceFunc, webView, @selector(webView:didParseSource:fromURL:sourceId:forWebFrame:), nsSource, [nsURL absoluteString], sourceProvider->asID(), webFrame);
        }
    } else {
        NSDictionary *info;
        if (errorMsg.isEmpty()) {
            info = @{
                WebScriptErrorLineNumberKey: @(errorLine),
            };
        } else {
            info = @{
                WebScriptErrorDescriptionKey: (NSString *)errorMsg,
                WebScriptErrorLineNumberKey: @(errorLine),
            };
        }
        auto error = adoptNS([[NSError alloc] initWithDomain:WebScriptErrorDomain code:WebScriptGeneralErrorCode userInfo:info]);

        if (implementations->failedToParseSourceFunc)
            CallScriptDebugDelegate(implementations->failedToParseSourceFunc, webView, @selector(webView:failedToParseSource:baseLineNumber:fromURL:withError:forWebFrame:), nsSource, firstLine, nsURL, error.get(), webFrame);
    }

    m_callingDelegate = false;
}

void WebScriptDebugger::handlePause(JSC::JSGlobalObject* globalObject, Debugger::ReasonForPause reason)
{
    if (m_callingDelegate)
        return;

    if (reason != Debugger::PausedForException)
        return;

    m_callingDelegate = true;

    JSC::VM& vm = globalObject->vm();
    WebFrame *webFrame = toWebFrame(globalObject);
    WebView *webView = [webFrame webView];
    JSC::DebuggerCallFrame& debuggerCallFrame = currentDebuggerCallFrame();
    JSC::JSValue exceptionValue = currentException();
    String functionName = debuggerCallFrame.functionName(vm);
    RetainPtr<WebScriptCallFrame> webCallFrame = adoptNS([[WebScriptCallFrame alloc] _initWithGlobalObject:core(webFrame)->script().windowScriptObject() functionName:functionName exceptionValue:exceptionValue]);

    WebScriptDebugDelegateImplementationCache* cache = WebViewGetScriptDebugDelegateImplementations(webView);
    if (cache->exceptionWasRaisedFunc) {
        if (cache->exceptionWasRaisedExpectsHasHandlerFlag) {
            bool hasHandler = hasHandlerForExceptionCallback();
            CallScriptDebugDelegate(cache->exceptionWasRaisedFunc, webView, @selector(webView:exceptionWasRaised:hasHandler:sourceId:line:forWebFrame:), webCallFrame.get(), hasHandler, debuggerCallFrame.sourceID(), debuggerCallFrame.line(), webFrame);
        } else
            CallScriptDebugDelegate(cache->exceptionWasRaisedFunc, webView, @selector(webView:exceptionWasRaised:sourceId:line:forWebFrame:), webCallFrame.get(), debuggerCallFrame.sourceID(), debuggerCallFrame.line(), webFrame);
    }

    m_callingDelegate = false;
}
