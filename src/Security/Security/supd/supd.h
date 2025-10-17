/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#import "supdProtocol.h"
#import "trust/trustd/trustdFileLocations.h"

NS_ASSUME_NONNULL_BEGIN

@class SFAnalyticsSQLiteStore;

@interface SFAnalyticsClient: NSObject

/// Returns an analytics client with the given name, if one already exists, or
/// creates and returns a new client with the given path and analytics settings
/// if not.
+ (SFAnalyticsClient *)getSharedClientNamed:(NSString *)name
                      orCreateWithStorePath:(NSString *)storePath
                     requireDeviceAnalytics:(BOOL)requireDeviceAnalytics
                     requireiCloudAnalytics:(BOOL)requireiCloudAnalytics;

+ (void)clearSFAnalyticsClientGlobalCache; // only for clearing cache by testing

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

//// Creates a client with a new store, name, and analytics settings.
- (instancetype)initWithStorePath:(NSString*)path
                             name:(NSString*)name
           requireDeviceAnalytics:(BOOL)requireDeviceAnalytics
           requireiCloudAnalytics:(BOOL)requireiCloudAnalytics;

/// Creates a client with the same underlying store and dispatch queue as an
/// existing client, but with a new name and analytics settings.
- (instancetype)initFromExistingClient:(SFAnalyticsClient *)client
                                  name:(NSString*)name
                requireDeviceAnalytics:(BOOL)requireDeviceAnalytics
                requireiCloudAnalytics:(BOOL)requireiCloudAnalytics;

/// Calls the given block with the underlying store.
///
/// It will open and close the store each time it's used.
///
/// Safety: The block must not re-entrantly call `-withStore:` on this client
/// (this will deadlock), and must not retain the store after returning (this
/// isn't thread-safe).
- (void)withStore:(void (^ NS_NOESCAPE)(SFAnalyticsSQLiteStore *store))block;

@property (readonly, nonatomic) NSString* storePath;
@property (readonly, nonatomic) NSString* name;
@property (readonly, nonatomic) BOOL requireDeviceAnalytics;
@property (readonly, nonatomic) BOOL requireiCloudAnalytics;

@end

@interface SFAnalyticsTopic : NSObject <NSURLSessionDelegate>
@property NSString* splunkTopicName;
@property NSURL* splunkBagURL;
@property NSString *internalTopicName;
@property NSUInteger uploadSizeLimit;

@property BOOL allowHTTPSplunkServerForTests;
@property BOOL allowInsecureSplunkCert;

@property NSArray<SFAnalyticsClient*>* topicClients;

// --------------------------------
// Things below are for unit testing
- (instancetype)initWithDictionary:(NSDictionary *)dictionary name:(NSString *)topicName samplingRates:(NSDictionary *)rates;
- (BOOL)haveEligibleClients;
- (NSArray<NSDictionary *> *_Nullable)createChunkedLoggingJSON:(NSArray<NSDictionary *> *)healthSummaries failures:(NSArray<NSDictionary *> *)failures error:(NSError **)error;
- (NSArray<NSArray *> *_Nullable)chunkFailureSet:(size_t)sizeCapacity events:(NSArray<NSDictionary *> *)events error:(NSError **)error;
- (size_t)serializedEventSize:(NSObject *)event error:(NSError**)error;
- (BOOL)ckDeviceAccountApprovedTopic:(NSString *)topic;


- (NSMutableDictionary*_Nullable)healthSummaryWithName:(SFAnalyticsClient*)client
                                                 store:(SFAnalyticsSQLiteStore*)store
                                                  uuid:(NSUUID *_Nullable)uuid
                                             timestamp:(NSNumber*_Nullable)timestamp
                                        lastUploadTime:(NSNumber*_Nullable)lastUploadTime;
- (NSDictionary *_Nullable)appleInternalStatus;

- (NSData *_Nullable)applyFilterLogic:(NSData *)data linkedID:(NSString *)linkedUUID;

+ (NSString*)databasePathForCKKS;
+ (NSString*)databasePathForSOS;
+ (NSString*)databasePathForPCS;
+ (NSString*)databasePathForLocal;
+ (NSString*)databasePathForTrust;
+ (NSString*)databasePathForNetworking;
+ (NSString*)databasePathForCloudServices;
+ (NSString*)databasePathForTransparency;
+ (NSString*)databasePathForSWTransparency;

#if TARGET_OS_OSX
+ (NSString*)databasePathForRootTrust;
+ (NSString*)databasePathForRootNetworking;
#endif

@end

@interface SFAnalyticsReporter : NSObject
- (BOOL)saveReportNamed:(NSString *)fileName reportData:(NSData *)reportData;
- (BOOL)saveReportNamed:(NSString *)fileName intoFileHandle:(void(^)(NSFileHandle *fileHandle))block;
@end

extern NSString* const SupdErrorDomain;
typedef NS_ERROR_ENUM(SupdErrorDomain, SupdError) {
    SupdNoError = 0,
    SupdGenericError,
    SupdInvalidJSONError,
    SupdMissingParamError,
};

@interface supd : NSObject <supdProtocol, TrustdFileHelper_protocol>
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithConnection:(NSXPCConnection *_Nullable)connection;
- (void)performRegularlyScheduledUpload;

- (BOOL)filebasedUploadAnalytics:(BOOL)force error:(NSError**)error;

// --------------------------------
// Things below are for unit testing
@property (readonly) NSArray<SFAnalyticsTopic*>* analyticsTopics;
@property (readonly) SFAnalyticsReporter *reporter;
- (void)sendNotificationForOncePerReportSamplers;
- (instancetype)initWithConnection:(NSXPCConnection *)connection reporter:(SFAnalyticsReporter *)reporter;
+ (NSData *_Nullable)serializeLoggingEvent:(NSDictionary *)event
                                     error:(NSError **)error;
+ (void)writeURL:(NSURL *)url intoFileHandle:(NSFileHandle *)fileHandle;

@end

// --------------------------------
// Things below are for unit testing
extern BOOL runningTests;                // Do not use 'force' when obtaining logging json
extern BOOL deviceAnalyticsOverride;
extern BOOL deviceAnalyticsEnabled;
extern BOOL iCloudAnalyticsOverride;
extern BOOL iCloudAnalyticsEnabled;

NS_ASSUME_NONNULL_END
