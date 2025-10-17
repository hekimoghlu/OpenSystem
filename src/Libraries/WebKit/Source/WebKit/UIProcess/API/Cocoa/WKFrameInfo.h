/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#import <WebKit/WKSecurityOrigin.h>
#import <WebKit/WKWebView.h>

/*! A WKFrameInfo object contains information about a frame on a webpage.
 @discussion An instance of this class is a transient, data-only object;
 it does not uniquely identify a frame across multiple delegate method
 calls.
 */
NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKFrameInfo : NSObject <NSCopying>

/*! @abstract A Boolean value indicating whether the frame is the main frame
 or a subframe.
 */
@property (nonatomic, readonly, getter=isMainFrame) BOOL mainFrame;

/*! @abstract The frame's current request.
 */
@property (nonatomic, readonly, copy) NSURLRequest *request;

/*! @abstract The frame's current security origin.
 */
@property (nonatomic, readonly) WKSecurityOrigin *securityOrigin WK_API_AVAILABLE(macos(10.11), ios(9.0));

/*! @abstract The web view of the webpage that contains this frame.
 */
@property (nonatomic, readonly, weak) WKWebView *webView WK_API_AVAILABLE(macos(10.13), ios(11.0));

@end

NS_ASSUME_NONNULL_END
