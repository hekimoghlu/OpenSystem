/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#import <WebKitLegacy/WebPolicyDelegate.h>

@class WebHistoryItem;
@class WebPolicyDecisionListenerPrivate;

extern NSString *WebActionFormKey; // HTMLFormElement

typedef enum {
    WebNavigationTypePlugInRequest = WebNavigationTypeOther + 1
} WebExtraNavigationType;

@interface WebPolicyDecisionListener : NSObject <WebPolicyDecisionListener>
{
@private
    WebPolicyDecisionListenerPrivate *_private;
}
- (id)_initWithTarget:(id)target action:(SEL)action;
- (void)_invalidate;
@end

@interface NSObject (WebPolicyDelegatePrivate)
// Needed for <rdar://problem/3951283> can view pages from the back/forward cache that should be disallowed by Parental Controls
- (BOOL)webView:(WebView *)webView shouldGoToHistoryItem:(WebHistoryItem *)item;
#if TARGET_OS_IPHONE
- (BOOL)webView:(WebView *)webView shouldLoadMediaURL:(NSURL *)url inFrame:(WebFrame *)frame;
#endif
@end
