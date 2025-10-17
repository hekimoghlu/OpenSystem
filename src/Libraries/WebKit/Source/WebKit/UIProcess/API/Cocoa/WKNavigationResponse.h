/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

@class WKFrameInfo;

/*! Contains information about a navigation response, used for making policy decisions.
 */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKNavigationResponse : NSObject

/*! @abstract A Boolean value indicating whether the frame being navigated is the main frame.
 */
@property (nonatomic, readonly, getter=isForMainFrame) BOOL forMainFrame;

/*! @abstract The frame's response.
 */
@property (nonatomic, readonly, copy) NSURLResponse *response;

/*! @abstract A Boolean value indicating whether WebKit can display the response's MIME type natively.
 @discussion Allowing a navigation response with a MIME type that can't be shown will cause the navigation to fail.
 */
@property (nonatomic, readonly) BOOL canShowMIMEType;

@end

NS_ASSUME_NONNULL_END
