/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#ifndef WebDownload_h
#define WebDownload_h

#import <Foundation/Foundation.h>
#import <WebKitLegacy/WebKitAvailability.h>

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST || (defined(USE_APPLE_INTERNAL_SDK) && USE_APPLE_INTERNAL_SDK)
#import <Foundation/NSURLDownload.h>
#else
__attribute__((visibility("hidden")))
@interface NSURLDownload : NSObject
@end

@protocol NSURLDownloadDelegate;
#endif

#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKAppKitStubs.h>
#endif

#if !TARGET_OS_IPHONE
@class NSWindow;
#endif
@class WebDownloadInternal;

/*!
    @class WebDownload
    @discussion A WebDownload works just like an NSURLDownload, with
    one extra feature: if you do not implement the
    authentication-related delegate methods, it will automatically
    prompt for authentication using the standard WebKit authentication
    panel, as either a sheet or window. It provides no extra methods,
    but does have one additional delegate method.
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface WebDownload : NSURLDownload
{
@package
    WebDownloadInternal *_webInternal;
}

@end

/*!
    @protocol WebDownloadDelegate
    @discussion The WebDownloadDelegate delegate has one extra method used to choose
    the right window when automatically prompting with a sheet.
*/
WEBKIT_DEPRECATED_MAC(10_4, 10_14)
@protocol WebDownloadDelegate <NSURLDownloadDelegate>

@optional

/*!
    @method downloadWindowForAuthenticationSheet:
*/
#if TARGET_OS_IPHONE
- (WAKWindow *)downloadWindowForAuthenticationSheet:(WebDownload *)download;
#else
- (NSWindow *)downloadWindowForAuthenticationSheet:(WebDownload *)download;
#endif

@end

#endif /* WebDownload_h */
