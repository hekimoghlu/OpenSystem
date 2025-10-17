/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class WKScriptMessage;
@class WKUserContentController;

/*! A class conforming to the WKScriptMessageHandler protocol provides a
 method for receiving messages from JavaScript running in a webpage.
 */
WK_SWIFT_UI_ACTOR
@protocol WKScriptMessageHandler <NSObject>

@required

/*! @abstract Invoked when a script message is received from a webpage.
 @param userContentController The user content controller invoking the
 delegate method.
 @param message The script message received.
 */
- (void)userContentController:(WKUserContentController *)userContentController didReceiveScriptMessage:(WKScriptMessage *)message;

@end

NS_ASSUME_NONNULL_END
