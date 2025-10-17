/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#import <WebKit/WKDownloadDelegate.h>
#import <WebKit/WKFoundation.h>

@class WKDownload;

/* @enum _WKPlaceholderPolicy
 @abstract The policy for creating a placeholder file in the Downloads directory during downloads.
 @constant _WKPlaceholderPolicyDisable   Do not create a placeholder file.
 @constant _WKPlaceholderPolicyEnable    Create a placeholder file.
 */
typedef NS_ENUM(NSInteger, _WKPlaceholderPolicy) {
    _WKPlaceholderPolicyDisable,
    _WKPlaceholderPolicyEnable,
} WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
@protocol WKDownloadDelegatePrivate <WKDownloadDelegate>

@optional

/* @abstract Invoked when the download needs a placeholder policy from the client.
 @param download The download for which we need a placeholder policy
 @param completionHandler The completion handler that should be invoked with the chosen policy
 @discussion The placeholder policy specifies whether a placeholder file should be created in
 the Downloads directory when the download is in progress. If the client opts out of the
 placeholder feature, it can choose to provide a custom URL to publish progress against.
 This is useful if the client maintains it's own placeholder file.
 */
- (void)_download:(WKDownload *)download decidePlaceholderPolicy:(void (^)(_WKPlaceholderPolicy, NSURL *))completionHandler WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

/* @abstract Called when the download receives a placeholder URL
 @param download The download for which we received a placeholder URL
 @param completionHandler The completion handler that should be called by the client in response to this call. 
 @discussion The placeholder URL will normally refer to a file in the Downloads directory
 */
- (void)_download:(WKDownload *)download didReceivePlaceholderURL:(NSURL *)url completionHandler:(void (^)(void))completionHandler WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

/* @abstract Called when the download receives a final URL
 @param download The download for which we received a final URL
 @param url The URL of the final download location
 @discussion The final URL will normally refer to a file in the Downloads directory
 */
- (void)_download:(WKDownload *)download didReceiveFinalURL:(NSURL *)url WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

@end

NS_ASSUME_NONNULL_END
