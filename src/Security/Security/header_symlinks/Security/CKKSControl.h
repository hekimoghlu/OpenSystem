/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
// You must be 64-bit to use this class.
#if __OBJC2__

#import <Foundation/Foundation.h>
#import <Security/CKKSExternalTLKClient.h>

NS_ASSUME_NONNULL_BEGIN


typedef NS_ENUM(NSUInteger, CKKSKnownBadState) {
    CKKSKnownStatePossiblyGood = 0,  // State might be good: give your operation a shot!
    CKKSKnownStateTLKsMissing = 1,   // CKKS doesn't have the TLKs: your operation will likely not succeed
    CKKSKnownStateWaitForUnlock = 2, // CKKS has some important things to do, but the device is locked. Your operation will likely not succeed
    CKKSKnownStateWaitForOctagon = 3, // CKKS has important things to do, but Octagon hasn't done them yet. Your operation will likely not succeed
    CKKSKnownStateNoCloudKitAccount = 4, // The device isn't signed into CloudKit. Your operation will likely not succeed
};

@interface CKKSControl : NSObject

@property (readonly,assign) BOOL synchronous;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithConnection:(NSXPCConnection*)connection;

- (void)rpcStatus:(NSString* _Nullable)viewName
        fast:(BOOL)fast
        waitForNonTransientState:(dispatch_time_t)nonTransientStateTimeout
        reply:(void(^)(NSArray<NSDictionary*>* _Nullable result, NSError* _Nullable error))reply;
- (void)rpcResetLocal:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcResetCloudKit:(NSString* _Nullable)viewName reason:(NSString *)reason reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcResyncLocal:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcResync:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcFetchAndProcessChanges:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcFetchAndProcessChangesIfNoRecentFetch:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcFetchAndProcessClassAChanges:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcPushOutgoingChanges:(NSString* _Nullable)viewName reply:(void (^)(NSError* _Nullable error))reply;
- (void)rpcCKMetric:(NSString *)eventName attributes:(NSDictionary *)attributes reply:(void(^)(NSError* error))reply;

- (void)rpcPerformanceCounters:             (void(^)(NSDictionary <NSString *,NSNumber *> *,NSError*))reply;
- (void)rpcGetCKDeviceIDWithReply:          (void (^)(NSString* ckdeviceID))reply;

// convenience wrappers for rpcStatus:fast:waitForNonTransientState:reply:
- (void)rpcStatus:(NSString* _Nullable)viewName
            reply:(void (^)(NSArray<NSDictionary*>* _Nullable result, NSError* _Nullable error))reply;
- (void)rpcFastStatus:(NSString* _Nullable)viewName
                reply:(void (^)(NSArray<NSDictionary*>* _Nullable result, NSError* _Nullable error))reply;
- (void)rpcTLKMissing:(NSString* _Nullable)viewName reply:(void (^)(bool missing))reply;
- (void)rpcKnownBadState:(NSString* _Nullable)viewName reply:(void (^)(CKKSKnownBadState))reply;

- (void)proposeTLKForSEView:(NSString*)seViewName
                proposedTLK:(CKKSExternalKey *)proposedTLK
              wrappedOldTLK:(CKKSExternalKey * _Nullable)wrappedOldTLK
                  tlkShares:(NSArray<CKKSExternalTLKShare*>*)shares
                      reply:(void(^)(NSError* _Nullable error))reply;

/* This API will cause the device to check in with CloudKit to get the most-up-to-date version of things */
- (void)fetchSEViewKeyHierarchy:(NSString*)seViewName
                          reply:(void (^)(CKKSExternalKey* _Nullable currentTLK,
                                          NSArray<CKKSExternalKey*>* _Nullable pastTLKs,
                                          NSArray<CKKSExternalTLKShare*>* _Nullable currentTLKShares,
                                          NSError* _Nullable error))reply;

/* If forceFetch is YES, then this API will check in with CLoudKit to get the most up-to-date version of things.
   If forceFetch is NO, then this API will the locally cached state. It will not wait for any currently-occuring fetches to complete. */
- (void)fetchSEViewKeyHierarchy:(NSString*)seViewName
                     forceFetch:(BOOL)forceFetch
                          reply:(void (^)(CKKSExternalKey* _Nullable currentTLK,
                                          NSArray<CKKSExternalKey*>* _Nullable pastTLKs,
                                          NSArray<CKKSExternalTLKShare*>* _Nullable currentTLKShares,
                                          NSError* _Nullable error))reply;

- (void)modifyTLKSharesForSEView:(NSString*)seViewName
                          adding:(NSArray<CKKSExternalTLKShare*>*)sharesToAdd
                        deleting:(NSArray<CKKSExternalTLKShare*>*)sharesToDelete
                          reply:(void (^)(NSError* _Nullable error))reply;

- (void)deleteSEView:(NSString*)seViewName
               reply:(void (^)(NSError* _Nullable error))reply;

- (void)toggleHavoc:(void (^)(BOOL havoc, NSError* _Nullable error))reply;

- (void)pcsMirrorKeysForServices:(NSDictionary<NSNumber*,NSArray<NSData*>*>*)services
                           reply:(void (^)(NSDictionary<NSNumber*,NSArray<NSData*>*>* _Nullable result,
                                           NSError* _Nullable error))reply;

/* Indicates whether we've fetched and processed all items in the account for this view at initial sign in time.
 This should not be used to determine whether we are currently in sync with CloudKit, and should not be used in lieu of notifications */
- (void)initialSyncStatus:(NSString*)viewName
                    reply:(void(^)(BOOL result, NSError* _Nullable error))reply;

+ (CKKSControl* _Nullable)controlObject:(NSError* _Nullable __autoreleasing* _Nullable)error;
+ (CKKSControl* _Nullable)CKKSControlObject:(BOOL)sync error:(NSError* _Nullable __autoreleasing* _Nullable)error;

@end

NS_ASSUME_NONNULL_END
#endif  // __OBJC__
