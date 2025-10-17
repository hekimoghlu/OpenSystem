/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

@class WKContentWorld;
@class WKFrameInfo;
@class WKWebView;

/*! A WKScriptMessage object contains information about a message sent from
 a webpage.
 */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKScriptMessage : NSObject

/*! @abstract The body of the message.
 @discussion Allowed types are NSNumber, NSString, NSDate, NSArray,
 NSDictionary, and NSNull.
 */
@property (nonatomic, readonly, copy) id body;

/*! @abstract The web view sending the message. */
@property (nullable, nonatomic, readonly, weak) WKWebView *webView;

/*! @abstract The frame sending the message. */
@property (nonatomic, readonly, copy) WKFrameInfo *frameInfo;

/*! @abstract The name of the message handler to which the message is sent.
 */
@property (nonatomic, readonly, copy) NSString *name;

/*! @abstract The content world from which the message was sent. */
@property (nonatomic, readonly) WKContentWorld *world WK_API_AVAILABLE(macos(11.0), ios(14.0));

@end

NS_ASSUME_NONNULL_END
