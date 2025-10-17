/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#ifdef __OBJC__

#import <Foundation/Foundation.h>

@class CKKSExternalKey;
@class CKKSExternalTLKShare;

NS_ASSUME_NONNULL_BEGIN;

#define CKKSControlStatusDefaultNonTransientStateTimeout (1*NSEC_PER_SEC)

@protocol CKKSControlProtocol <NSObject>
- (void)performanceCounters:(void(^)(NSDictionary <NSString *, NSNumber *> * _Nullable))reply;
- (void)rpcResetLocal: (NSString* _Nullable)viewName reply: (void(^)(NSError* _Nullable result)) reply;

/**
 * Reset CloudKit zone with a caller provided reason, the reason will be logged in the operation group
 * name so that the reason for reset can be summarized server side.
 */
- (void)rpcResetCloudKit:(NSString* _Nullable)viewName reason:(NSString *)reason reply: (void(^)(NSError* _Nullable result)) reply;
- (void)rpcResync:(NSString* _Nullable)viewName reply: (void(^)(NSError* _Nullable result)) reply;
- (void)rpcResyncLocal:(NSString* _Nullable)viewName reply:(void(^)(NSError* _Nullable result))reply;

/**
 * Fetch status for the CKKS zones. If NULL is passed in a viewname, all zones are fetched.
 * If `fast` is `YES`, this call will avoid expensive operations (and thus not
 * report them), and also omit the global status.
 */
- (void)rpcStatus:(NSString* _Nullable)viewName fast:(BOOL)fast waitForNonTransientState:(dispatch_time_t)waitForTransientTimeout reply: (void(^)(NSArray<NSDictionary*>* _Nullable result, NSError* _Nullable error)) reply;

- (void)rpcFetchAndProcessChanges:(NSString* _Nullable)viewName classA:(bool)classAError onlyIfNoRecentFetch:(bool)onlyIfNoRecentFetch reply:(void(^)(NSError* _Nullable result))reply;
- (void)rpcPushOutgoingChanges:(NSString* _Nullable)viewName reply: (void(^)(NSError* _Nullable result)) reply;
- (void)rpcGetCKDeviceIDWithReply:(void (^)(NSString* _Nullable ckdeviceID))reply;
- (void)rpcCKMetric:(NSString *)eventName attributes:(NSDictionary *)attributes reply:(void(^)(NSError* _Nullable result)) reply;

- (void)proposeTLKForSEView:(NSString*)seViewName
                proposedTLK:(CKKSExternalKey *)proposedTLK
              wrappedOldTLK:(CKKSExternalKey * _Nullable)wrappedOldTLK
                  tlkShares:(NSArray<CKKSExternalTLKShare*>*)shares
                      reply:(void(^)(NSError* _Nullable error))reply;

- (void)fetchSEViewKeyHierarchy:(NSString*)seViewName
                     forceFetch:(BOOL)forceFetch
                          reply:(void (^)(CKKSExternalKey* _Nullable currentTLK,
                                          NSArray<CKKSExternalKey*>* _Nullable pastTLKs,
                                          NSArray<CKKSExternalTLKShare*>* _Nullable curentTLKShares,
                                          NSError* _Nullable error))reply;

- (void)modifyTLKSharesForSEView:(NSString*)seViewName
                          adding:(NSArray<CKKSExternalTLKShare*>*)sharesToAdd
                        deleting:(NSArray<CKKSExternalTLKShare*>*)sharesToDelete
                           reply:(void (^)(NSError* _Nullable error))reply;

- (void)deleteSEView:(NSString*)seViewName
               reply:(void (^)(NSError* _Nullable error))reply;

- (void)toggleHavoc:(void (^)(BOOL havoc, NSError* _Nullable error))reply;

- (void)pcsMirrorKeysForServices:(NSDictionary<NSNumber*,NSArray<NSData*>*>*)services reply:(void (^)(NSDictionary<NSNumber*,NSArray<NSData*>*>* _Nullable result, NSError* _Nullable error))reply;

- (void)initialSyncStatus:(NSString*)viewName
                    reply:(void(^)(BOOL finished, NSError* _Nullable error))reply;

@end

NSXPCInterface* CKKSSetupControlProtocol(NSXPCInterface* interface);

NS_ASSUME_NONNULL_END

#endif /* __OBJC__ */
