/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

@class WKBrowsingContextController;

/* Constants for policy action dictionaries */
WK_EXTERN NSString * const WKActionIsMainFrameKey;         // NSNumber (BOOL)
WK_EXTERN NSString * const WKActionMouseButtonKey;         // NSNumber (0 for left button, 1 for middle button, 2 for right button)
WK_EXTERN NSString * const WKActionModifierFlagsKey;       // NSNumber (unsigned)
WK_EXTERN NSString * const WKActionOriginalURLRequestKey;  // NSURLRequest
WK_EXTERN NSString * const WKActionURLRequestKey;          // NSURLRequest
WK_EXTERN NSString * const WKActionURLResponseKey;         // NSURLResponse
WK_EXTERN NSString * const WKActionFrameNameKey;           // NSString
WK_EXTERN NSString * const WKActionOriginatingFrameURLKey; // NSURL
WK_EXTERN NSString * const WKActionCanShowMIMETypeKey;     // NSNumber (BOOL)

typedef NS_ENUM(NSUInteger, WKPolicyDecision) {
    WKPolicyDecisionCancel,
    WKPolicyDecisionAllow,
    WKPolicyDecisionBecomeDownload
};

typedef void (^WKPolicyDecisionHandler)(WKPolicyDecision);

WK_CLASS_DEPRECATED_WITH_REPLACEMENT("WKNavigationDelegate", macos(10.10, 10.14.4), ios(8.0, 12.2))
@protocol WKBrowsingContextPolicyDelegate <NSObject>
@optional

- (void)browsingContextController:(WKBrowsingContextController *)sender decidePolicyForNavigationAction:(NSDictionary *)actionInformation decisionHandler:(WKPolicyDecisionHandler)decisionHandler;
- (void)browsingContextController:(WKBrowsingContextController *)sender decidePolicyForNewWindowAction:(NSDictionary *)actionInformation decisionHandler:(WKPolicyDecisionHandler)decisionHandler;
- (void)browsingContextController:(WKBrowsingContextController *)sender decidePolicyForResponseAction:(NSDictionary *)actionInformation decisionHandler:(WKPolicyDecisionHandler)decisionHandler;

@end
