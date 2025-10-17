/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
#import <Foundation/Foundation.h>

typedef intptr_t WebSourceId;

@class WebView;
@class WebFrame;
@class WebScriptCallFrame;
@class WebScriptCallFramePrivate;
@class WebScriptObject;

extern NSString * const WebScriptErrorDomain;
extern NSString * const WebScriptErrorDescriptionKey;
extern NSString * const WebScriptErrorLineNumberKey;

enum {
    WebScriptGeneralErrorCode = -100
};

// WebScriptDebugDelegate messages

@interface NSObject (WebScriptDebugDelegate)

// some source was parsed, establishing a "source ID" (>= 0) for future reference
// this delegate method is deprecated, please switch to the new version below
- (void)webView:(WebView *)webView       didParseSource:(NSString *)source
                                                fromURL:(NSString *)url
                                               sourceId:(WebSourceId)sid
                                            forWebFrame:(WebFrame *)webFrame;

// some source was parsed, establishing a "source ID" (>= 0) for future reference
- (void)webView:(WebView *)webView       didParseSource:(NSString *)source
                                         baseLineNumber:(NSUInteger)lineNumber
                                                fromURL:(NSURL *)url
                                               sourceId:(WebSourceId)sid
                                            forWebFrame:(WebFrame *)webFrame;

// some source failed to parse
- (void)webView:(WebView *)webView  failedToParseSource:(NSString *)source
                                         baseLineNumber:(NSUInteger)lineNumber
                                                fromURL:(NSURL *)url
                                              withError:(NSError *)error
                                            forWebFrame:(WebFrame *)webFrame;

// exception is being thrown
- (void)webView:(WebView *)webView   exceptionWasRaised:(WebScriptCallFrame *)frame
                                             hasHandler:(BOOL)hasHandler
                                               sourceId:(WebSourceId)sid
                                                   line:(int)lineno
                                            forWebFrame:(WebFrame *)webFrame;

// exception is being thrown (deprecated old version; called only if new version is not implemented)
- (void)webView:(WebView *)webView   exceptionWasRaised:(WebScriptCallFrame *)frame
                                               sourceId:(WebSourceId)sid
                                                   line:(int)lineno
                                            forWebFrame:(WebFrame *)webFrame;

@end



// WebScriptCallFrame interface
//
// These objects are passed as arguments to the debug delegate.

@interface WebScriptCallFrame : NSObject
{
@private
    WebScriptCallFramePrivate* _private;
    id                         _userInfo;
}

// associate user info with frame
- (void)setUserInfo:(id)userInfo;

// retrieve user info
- (id)userInfo;

// get name of function (if available) or nil
- (NSString *)functionName;

// get pending exception (if any) or nil
- (id)exception;

@end
