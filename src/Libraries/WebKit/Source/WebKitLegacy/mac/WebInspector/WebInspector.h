/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#import <Foundation/NSObject.h>

extern NSString *WebInspectorDidStartSearchingForNode;
extern NSString *WebInspectorDidStopSearchingForNode;

@class WebView;
@class WebInspectorFrontend;

@interface WebInspector : NSObject
{
@private
    WebView *_inspectedWebView;
    WebInspectorFrontend *_frontend;
}
- (id)initWithInspectedWebView:(WebView *)inspectedWebView;
- (void)inspectedWebViewClosed;
- (void)show:(id)sender;
- (void)showConsole:(id)sender;
- (void)close:(id)sender;
- (void)attach:(id)sender;
- (void)detach:(id)sender;

- (BOOL)isDebuggingJavaScript;
- (void)toggleDebuggingJavaScript:(id)sender;
- (void)startDebuggingJavaScript:(id)sender;
- (void)stopDebuggingJavaScript:(id)sender;

- (BOOL)isJavaScriptProfilingEnabled;
- (void)setJavaScriptProfilingEnabled:(BOOL)enabled;
- (BOOL)isTimelineProfilingEnabled;
- (void)setTimelineProfilingEnabled:(BOOL)enabled;

- (BOOL)isProfilingJavaScript;
- (void)toggleProfilingJavaScript:(id)sender;
- (void)startProfilingJavaScript:(id)sender;
- (void)stopProfilingJavaScript:(id)sender;

@property (nonatomic, readonly, getter=isOpen) BOOL open;
@end
