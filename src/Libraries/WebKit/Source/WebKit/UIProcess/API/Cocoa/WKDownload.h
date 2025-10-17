/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#import <WebKit/WKFoundation.h>

@class WKFrameInfo;
@class WKWebView;
@protocol NSProgressReporting;
@protocol WKDownloadDelegate;

NS_ASSUME_NONNULL_BEGIN

WK_CLASS_AVAILABLE(macos(11.3), ios(14.5))
WK_SWIFT_UI_ACTOR
@interface WKDownload : NSObject<NSProgressReporting>

/* @abstract The request used to initiate this download.
  @discussion If the original request redirected to a different URL, originalRequest
  will be unchanged after the download follows the redirect.
 */
@property (nonatomic, readonly, nullable) NSURLRequest *originalRequest;

/* @abstract The web view that originated this download. */
@property (nonatomic, readonly, weak) WKWebView *webView;

/* @abstract The delegate that receives progress updates for this download. */
@property (nonatomic, weak) id <WKDownloadDelegate> delegate;

/* @abstract A boolean value indicating whether this download was initiated by the user. */
@property (nonatomic, readonly, getter=isUserInitiated) BOOL userInitiated WK_API_AVAILABLE(macos(15.2), ios(18.2));

/* @abstract The frame that originated this download. */
@property (nonatomic, readonly) WKFrameInfo *originatingFrame WK_API_AVAILABLE(macos(15.2), ios(18.2));

/* @abstract Cancel the download.
 @param completionHandler A block to invoke when cancellation is finished.
 @discussion To attempt to resume the download, call WKWebView resumeDownloadFromResumeData: with the data given to the completionHandler.
 If no resume attempt is possible with this server, completionHandler will be called with nil.
 */
- (void)cancel:(WK_SWIFT_UI_ACTOR void(^ _Nullable)(NSData * _Nullable resumeData))completionHandler;

@end

NS_ASSUME_NONNULL_END
