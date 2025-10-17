/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#if USE(APPLE_INTERNAL_SDK)

#import <NetworkExtension/NEFilterSource.h>

#else

typedef NS_ENUM(NSInteger, NEFilterSourceStatus) {
    NEFilterSourceStatusPass = 1,
    NEFilterSourceStatusBlock = 2,
    NEFilterSourceStatusNeedsMoreData = 3,
    NEFilterSourceStatusError = 4,
    NEFilterSourceStatusWhitelisted = 5,
    NEFilterSourceStatusBlacklisted = 6,
};

typedef NS_ENUM(NSInteger, NEFilterSourceDirection) {
    NEFilterSourceDirectionOutbound = 1,
    NEFilterSourceDirectionInbound = 2,
};

@interface NEFilterSource : NSObject
@end

@interface NEFilterSource (WKLegacyDetails)
+ (BOOL)filterRequired;
- (id)initWithURL:(NSURL *)url direction:(NEFilterSourceDirection)direction socketIdentifier:(uint64_t)socketIdentifier;
- (void)addData:(NSData *)data withCompletionQueue:(dispatch_queue_t)queue completionHandler:(void (^)(NEFilterSourceStatus, NSData *))completionHandler;
- (void)dataCompleteWithCompletionQueue:(dispatch_queue_t)queue completionHandler:(void (^)(NEFilterSourceStatus, NSData *))completionHandler;
@property (readonly) NEFilterSourceStatus status;
@property (readonly) NSURL *url;
@property (readonly) NEFilterSourceDirection direction;
@property (readonly) uint64_t socketIdentifier;
@end

#define NEFilterSourceOptionsPageData @"PageData"
#define NEFilterSourceOptionsRedirectURL @"RedirectURL"

typedef void (^NEFilterSourceDecisionHandler)(NEFilterSourceStatus, NSDictionary *);

@interface NEFilterSource (WKModernDetails)
- (id)initWithDecisionQueue:(dispatch_queue_t)queue;
- (void)willSendRequest:(NSURLRequest *)request decisionHandler:(NEFilterSourceDecisionHandler)decisionHandler;
- (void)receivedResponse:(NSURLResponse *)response decisionHandler:(NEFilterSourceDecisionHandler)decisionHandler;
- (void)receivedData:(NSData *)data decisionHandler:(NEFilterSourceDecisionHandler)decisionHandler;
- (void)finishedLoadingWithDecisionHandler:(NEFilterSourceDecisionHandler)decisionHandler;
- (void)remediateWithDecisionHandler:(NEFilterSourceDecisionHandler)decisionHandler;
@property (copy) NSString *sourceAppIdentifier;
@property (assign) pid_t sourceAppPid;
@end

#endif // !USE(APPLE_INTERNAL_SDK)

@interface NEFilterSource (Delegation)
+ (void)setDelegation:(audit_token_t *)audit_token;
@end
