/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
@class WKDownload;

WK_CLASS_DEPRECATED_WITH_REPLACEMENT("WKDownload", macos(10.10, 12.0), ios(8.0, 15.0))
@interface _WKDownload : NSObject <NSCopying>

+ (instancetype)downloadWithDownload:(WKDownload *)download WK_API_AVAILABLE(macos(12.0), ios(15.0));

- (void)cancel;
- (void)publishProgressAtURL:(NSURL *)URL WK_API_AVAILABLE(macos(10.14.4), ios(12.2));

@property (nonatomic, readonly) NSURLRequest *request;
@property (nonatomic, readonly, weak) WKWebView *originatingWebView;
@property (nonatomic, readonly, copy) NSArray<NSURL *> *redirectChain WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
@property (nonatomic, readonly) BOOL wasUserInitiated WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
@property (nonatomic, readonly) NSData *resumeData WK_API_AVAILABLE(macos(10.14.4), ios(12.2));
@property (nonatomic, readonly) WKFrameInfo *originatingFrame WK_API_AVAILABLE(macos(10.15.4), ios(13.4));

@end
