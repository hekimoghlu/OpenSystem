/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#import <WebKit/NSAttributedString.h>

@class WKNavigation;
@class WKWebView;

NS_ASSUME_NONNULL_BEGIN

/*!
 @abstract Indicates additional local paths WebKit can read from when loading content.
 The value is an NSArray containing one or more NSURLs.
*/
WK_EXTERN NSAttributedStringDocumentReadingOptionKey const _WKReadAccessFileURLsOption
    NS_SWIFT_NAME(readAccessPaths) WK_API_AVAILABLE(macos(13.1), ios(16.2));

/*!
 @abstract Whether to allow loads over the network (including subresources).
 The value is an NSNumber, which is interpreted as a BOOL.
*/
WK_EXTERN NSAttributedStringDocumentReadingOptionKey const _WKAllowNetworkLoadsOption
    NS_SWIFT_NAME(allowNetworkLoads) WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

/*!
 @abstract Bundle identifier of the application to which network activity is attributed.
 The value is an NSString.
*/
WK_EXTERN NSAttributedStringDocumentReadingOptionKey const _WKSourceApplicationBundleIdentifierOption
    NS_SWIFT_NAME(sourceApplicationBundleIdentifier) WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

/*!
 @discussion Private extension of @link //apple_ref/occ/NSAttributedString NSAttributedString @/link to
 translate HTML content into attributed strings using WebKit.
 */
@interface NSAttributedString (WKPrivate)

/*!
 @abstract Converts the contents loaded by a content loader block into an attributed string.
 @param options Document attributes for interpreting the document contents.
 NSTextSizeMultiplierDocumentOption, and NSTimeoutDocumentOption are supported option keys.
 @param contentLoader A block to invoke when content needs to be loaded in the supplied
 @link WKWebView @/link. A @link WKNavigation @/link for the main frame must be returned.
 @param completionHandler A block to invoke when the translation completes or fails.
 @discussion The completionHandler is passed the attributed string result along with any
 document-level attributes, or an error.
 */
+ (void)_loadFromHTMLWithOptions:(NSDictionary<NSAttributedStringDocumentReadingOptionKey, id> *)options contentLoader:(WKNavigation *(^)(WKWebView *))loadWebContent completionHandler:(NSAttributedStringCompletionHandler)completionHandler;

@end

NS_ASSUME_NONNULL_END
