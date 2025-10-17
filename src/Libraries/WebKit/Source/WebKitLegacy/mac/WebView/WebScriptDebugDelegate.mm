/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#import "WebScriptDebugDelegate.h"
#import "WebDataSource.h"
#import "WebDataSourceInternal.h"
#import "WebFrameInternal.h"
#import "WebScriptDebugger.h"
#import "WebViewInternal.h"
#import <JavaScriptCore/CallFrame.h>
#import <JavaScriptCore/Completion.h>
#import <JavaScriptCore/Debugger.h>
#import <JavaScriptCore/JSFunction.h>
#import <JavaScriptCore/JSGlobalObject.h>
#import <JavaScriptCore/JSLock.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/ScriptController.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/runtime_root.h>

// FIXME: these error strings should be public for future use by WebScriptObject and in WebScriptObject.h
NSString * const WebScriptErrorDomain = @"WebScriptErrorDomain";
NSString * const WebScriptErrorDescriptionKey = @"WebScriptErrorDescription";
NSString * const WebScriptErrorLineNumberKey = @"WebScriptErrorLineNumber";

@interface WebScriptCallFrame (WebScriptDebugDelegateInternalForDelegate)

- (id)_convertValueToObjcValue:(JSC::JSValue)value;

@end

@interface WebScriptCallFramePrivate : NSObject {
@public
    WebScriptObject        *globalObject;   // the global object's proxy (not retained)
    String functionName;
    JSC::JSValue exceptionValue;
}
@end

@implementation WebScriptCallFramePrivate
- (void)dealloc
{
    [super dealloc];
}
@end

// WebScriptCallFrame
//
// One of these is created to represent each stack frame.  Additionally, there is a "global"
// frame to represent the outermost scope.  This global frame is always the last frame in
// the chain of callers.
//
// The delegate can assign a "wrapper" to each frame object so it can relay calls through its
// own exported interface.  This class is private to WebCore (and the delegate).

@implementation WebScriptCallFrame (WebScriptDebugDelegateInternal)

- (WebScriptCallFrame *)_initWithGlobalObject:(WebScriptObject *)globalObj functionName:(String)functionName exceptionValue:(JSC::JSValue)exceptionValue
{
    if ((self = [super init])) {
        _private = [[WebScriptCallFramePrivate alloc] init];
        _private->globalObject = globalObj;
        _private->functionName = functionName;
        _private->exceptionValue = exceptionValue;
    }
    return self;
}

- (id)_convertValueToObjcValue:(JSC::JSValue)value
{
    if (!value)
        return nil;

    WebScriptObject *globalObject = _private->globalObject;
    if (value == [globalObject _imp])
        return globalObject;

    JSC::Bindings::RootObject* root1 = [globalObject _originRootObject];
    if (!root1)
        return nil;

    JSC::Bindings::RootObject* root2 = [globalObject _rootObject];
    if (!root2)
        return nil;

    return [WebScriptObject _convertValueToObjcValue:value originRootObject:root1 rootObject:root2];
}

@end



@implementation WebScriptCallFrame

- (void) dealloc
{
    [_userInfo release];
    [_private release];
    [super dealloc];
}

- (void)setUserInfo:(id)userInfo
{
    if (userInfo != _userInfo) {
        [_userInfo release];
        _userInfo = [userInfo retain];
    }
}

- (id)userInfo
{
    return _userInfo;
}

// Returns the name of the function for this frame, if available.
// Returns nil for anonymous functions and for the global frame.

- (NSString *)functionName
{
    if (_private->functionName.isEmpty())
        return nil;

    String functionName = _private->functionName;
    return nsStringNilIfEmpty(functionName);
}

// Returns the pending exception for this frame (nil if none).

- (id)exception
{
    if (!_private->exceptionValue)
        return nil;

    JSC::JSValue exception = _private->exceptionValue;
    return exception ? [self _convertValueToObjcValue:exception] : nil;
}

@end
