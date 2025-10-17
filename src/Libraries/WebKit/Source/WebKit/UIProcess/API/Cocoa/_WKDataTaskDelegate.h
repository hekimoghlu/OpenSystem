/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

@class _WKDataTask;

typedef NS_ENUM(NSInteger, _WKDataTaskRedirectPolicy) {
    _WKDataTaskRedirectPolicyCancel,
    _WKDataTaskRedirectPolicyAllow,
} WK_API_AVAILABLE(macos(13.0), ios(16.0));

typedef NS_ENUM(NSInteger, _WKDataTaskResponsePolicy) {
    _WKDataTaskResponsePolicyCancel,
    _WKDataTaskResponsePolicyAllow,
} WK_API_AVAILABLE(macos(13.0), ios(16.0));

NS_ASSUME_NONNULL_BEGIN

@protocol _WKDataTaskDelegate <NSObject>

@optional

- (void)dataTask:(_WKDataTask *)dataTask didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition, NSURLCredential * _Nullable))completionHandler;
- (void)dataTask:(_WKDataTask *)dataTask willPerformHTTPRedirection:(NSHTTPURLResponse *)response newRequest:(NSURLRequest *)request decisionHandler:(void (^)(_WKDataTaskRedirectPolicy))decisionHandler;
- (void)dataTask:(_WKDataTask *)dataTask didReceiveResponse:(NSURLResponse *)response decisionHandler:(void (^)(_WKDataTaskResponsePolicy))decisionHandler;
- (void)dataTask:(_WKDataTask *)dataTask didReceiveData:(NSData *)data;
- (void)dataTask:(_WKDataTask *)dataTask didCompleteWithError:(nullable NSError *)error;

@end

NS_ASSUME_NONNULL_END
