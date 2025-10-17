/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#import <WebKit/WKFoundation.h>

@class WKProcessPool;
@class _WKAutomationSessionConfiguration;

@protocol _WKAutomationDelegate <NSObject>
@optional
- (BOOL)_processPoolAllowsRemoteAutomation:(WKProcessPool *)processPool WK_API_AVAILABLE(macos(10.13), ios(11.0));
- (void)_processPool:(WKProcessPool *)processPool didRequestAutomationSessionWithIdentifier:(NSString *)identifier configuration:(_WKAutomationSessionConfiguration *)configuration WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

- (NSString *)_processPoolBrowserNameForAutomation:(WKProcessPool *)processPool WK_API_AVAILABLE(macos(10.14), ios(12.0));
- (NSString *)_processPoolBrowserVersionForAutomation:(WKProcessPool *)processPool WK_API_AVAILABLE(macos(10.14), ios(12.0));

- (void)_processPoolDidRequestInspectorDebuggablesToWakeUp:(WKProcessPool *)processPool WK_API_AVAILABLE(macos(12.0), ios(15.0));
@end
