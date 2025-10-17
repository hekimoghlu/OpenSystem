/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

@class _WKDownload;

@protocol _WKDownloadDelegate <NSObject>
@optional
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
- (void)_downloadDidStart:(_WKDownload *)download;
- (void)_download:(_WKDownload *)download didReceiveServerRedirectToURL:(NSURL *)url WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (void)_download:(_WKDownload *)download didReceiveResponse:(NSURLResponse *)response;
- (void)_download:(_WKDownload *)download didReceiveData:(uint64_t)length;
- (void)_download:(_WKDownload *)download didWriteData:(uint64_t)bytesWritten totalBytesWritten:(uint64_t)totalBytesWritten totalBytesExpectedToWrite:(uint64_t)totalBytesExpectedToWrite WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (void)_download:(_WKDownload *)download decideDestinationWithSuggestedFilename:(NSString *)filename completionHandler:(void (^)(BOOL allowOverwrite, NSString *destination))completionHandler;
- (void)_downloadDidFinish:(_WKDownload *)download;
- (void)_download:(_WKDownload *)download didFailWithError:(NSError *)error;
- (void)_downloadDidCancel:(_WKDownload *)download;
- (void)_download:(_WKDownload *)download didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition, NSURLCredential*))completionHandler WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (void)_download:(_WKDownload *)download didCreateDestination:(NSString *)destination WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (void)_downloadProcessDidCrash:(_WKDownload *)download WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (BOOL)_download:(_WKDownload *)download shouldDecodeSourceDataOfMIMEType:(NSString *)MIMEType WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

- (NSString *)_download:(_WKDownload *)download decideDestinationWithSuggestedFilename:(NSString *)filename allowOverwrite:(BOOL *)allowOverwrite WK_API_DEPRECATED_WITH_REPLACEMENT("_download:decideDestinationWithSuggestedFilename:completionHandler:", macos(10.10, 10.13.4), ios(8.0, 11.3));
#pragma clang diagnostic pop
@end
