/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

@class NSURLRequest;
@class NSString;
@class WebFrame;
@class WebView;
@class WebPolicyDecisionListener;

@protocol WebPluginContainerCheckController <NSObject>
- (void)_webPluginContainerCancelCheckIfAllowedToLoadRequest:(id)checkIdentifier;
- (WebFrame *)webFrame;
- (WebView *)webView;
@end

@interface WebPluginContainerCheck : NSObject
{
    NSURLRequest *_request;
    NSString *_target;
    id <WebPluginContainerCheckController> _controller;
    id _resultObject;
    SEL _resultSelector;
    id _contextInfo;
    BOOL _done;
    WebPolicyDecisionListener *_listener;
}

+ (id)checkWithRequest:(NSURLRequest *)request target:(NSString *)target resultObject:(id)obj selector:(SEL)selector controller:(id <WebPluginContainerCheckController>)controller contextInfo:(id)/*optional*/contextInfo; 
- (void)start;
- (void)cancel;
- (id)contextInfo;

@end
