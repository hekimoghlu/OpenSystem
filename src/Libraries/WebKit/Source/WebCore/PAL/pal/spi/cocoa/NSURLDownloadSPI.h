/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#if PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)

#import <Foundation/NSURLDownload.h>

#if USE(APPLE_INTERNAL_SDK)
#import <Foundation/NSURLDownloadPrivate.h>
#endif

#else

@class NSString;
@class NSURLAuthenticationChallenge;
@class NSURLDownload;
@class NSURLProtectionSpace;
@class NSURLRequest;
@class NSURLResponse;

#ifndef WebDownload_h
/* Also defined in <WebKit/WebDownload.h>. */
@interface NSURLDownload : NSObject
@end
#endif

@protocol NSURLDownloadDelegate <NSObject>
@optional
- (void)downloadDidBegin:(NSURLDownload *)download;
- (NSURLRequest *)download:(NSURLDownload *)download willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse;
- (BOOL)download:(NSURLDownload *)connection canAuthenticateAgainstProtectionSpace:(NSURLProtectionSpace *)protectionSpace;
- (void)download:(NSURLDownload *)download didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge;
- (BOOL)downloadShouldUseCredentialStorage:(NSURLDownload *)download;
- (void)download:(NSURLDownload *)download didReceiveResponse:(NSURLResponse *)response;
- (void)download:(NSURLDownload *)download willResumeWithResponse:(NSURLResponse *)response fromByte:(long long)startingByte;
- (void)download:(NSURLDownload *)download didReceiveDataOfLength:(NSUInteger)length;
- (BOOL)download:(NSURLDownload *)download shouldDecodeSourceDataOfMIMEType:(NSString *)encodingType;
- (void)download:(NSURLDownload *)download decideDestinationWithSuggestedFilename:(NSString *)filename;
- (void)download:(NSURLDownload *)download didCreateDestination:(NSString *)path;
- (void)downloadDidFinish:(NSURLDownload *)download;
- (void)download:(NSURLDownload *)download didFailWithError:(NSError *)error;
@end

@interface NSURLDownload ()
- (instancetype)initWithRequest:(NSURLRequest *)request delegate:(id <NSURLDownloadDelegate>)delegate;
- (instancetype)initWithResumeData:(NSData *)resumeData delegate:(id <NSURLDownloadDelegate>)delegate path:(NSString *)path;
- (void)cancel;
- (void)setDestination:(NSString *)path allowOverwrite:(BOOL)allowOverwrite;
@property (readonly, copy) NSURLRequest *request;
@property (readonly, copy) NSData *resumeData;
@property BOOL deletesFileUponFailure;
@end

#endif

#if !USE(APPLE_INTERNAL_SDK)

@class NSURLConnectionDelegateProxy;

@interface NSURLDownload ()
+ (id)_downloadWithRequest:(NSURLRequest *)request delegate:(id)delegate directory:(NSString *)directory;
+ (id)_downloadWithLoadingConnection:(NSURLConnection *)connection request:(NSURLRequest *)request response:(NSURLResponse *)response delegate:(id)delegate proxy:(id)proxy;
- (id)_initWithLoadingConnection:(NSURLConnection *)connection request:(NSURLRequest *)request response:(NSURLResponse *)response delegate:(id)delegate proxy:(NSURLConnectionDelegateProxy *)proxy;
- (id)_initWithRequest:(NSURLRequest *)request delegate:(id)delegate directory:(NSString *)directory;
@end

#endif
