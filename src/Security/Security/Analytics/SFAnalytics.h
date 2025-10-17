/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#if __OBJC2__
#ifndef SFAnalytics_h
#define SFAnalytics_h

#import <Foundation/Foundation.h>
#import <Security/SFAnalyticsDefines.h>
#import <Security/SFAnalyticsSampler.h>
#import <Security/SFAnalyticsMultiSampler.h>
#import <Security/SFAnalyticsActivityTracker.h>

NS_ASSUME_NONNULL_BEGIN

@class SecLaunchSequence;

extern NSString *kSecSFAErrorDomain;
typedef NS_ERROR_ENUM(kSecSFAErrorDomain, kSecSFAErrorCode) {
    kSecSFAErrorsRulesMissing = 1,
    kSecSFAErrorTypeMissing,
    kSecSFAErrorRulesInvalidType,
    kSecSFAErrorMatchMissing,
    kSecSFAErrorSecondInvalid,
    kSecSFAErrorActionInvalidType,
    kSecSFAErrorActionInvalid,
    kSecSFAErrorRadarInvalidType,
    kSecSFAErrorTTRAttributeInvalidType,
    kSecSFAErrorABCAttributeInvalidType,
    kSecSFAErrorUnknownAction,
    kSecSFAErrorFailedToEncodeMatchStructure,
    kSecSFAErrorsPropsMissing,
    kSecSFAErrorPropsInvalidType,
    kSecSFAErrorVersionMismatch,
    kSecSFAErrorVersionMissing,
};


// this sampling interval will cause the sampler to run only at data reporting time
extern const NSTimeInterval SFAnalyticsSamplerIntervalOncePerReport;

typedef NS_ENUM(uint32_t, SFAnalyticsTimestampBucket) {
    SFAnalyticsTimestampBucketSecond = 0,
    SFAnalyticsTimestampBucketMinute = 1,
    SFAnalyticsTimestampBucketHour = 2,
};

typedef NS_OPTIONS(uint32_t, SFAnalyticsMetricsHookActions) {
    SFAnalyticsMetricsHookNoAction = 0,
    SFAnalyticsMetricsHookExcludeEvent = 1,
    SFAnalyticsMetricsHookExcludeCount = 2,
};

typedef SFAnalyticsMetricsHookActions(^SFAnalyticsMetricsHook)(NSString* eventName,
                                                               SFAnalyticsEventClass eventClass,
                                                               NSDictionary* attributes,
                                                               SFAnalyticsTimestampBucket timestampBucket);

@protocol SFAnalyticsProtocol <NSObject>
+ (id<SFAnalyticsProtocol> _Nullable)logger;

- (void)logResultForEvent:(NSString*)eventName
              hardFailure:(bool)hardFailure
                   result:(NSError* _Nullable)eventResultError;
- (void)logResultForEvent:(NSString*)eventName
              hardFailure:(bool)hardFailure
                   result:(NSError* _Nullable)eventResultError
           withAttributes:(NSDictionary* _Nullable)attributes;

- (SFAnalyticsMultiSampler* _Nullable)AddMultiSamplerForName:(NSString *)samplerName
                                            withTimeInterval:(NSTimeInterval)timeInterval
                                                       block:(NSDictionary<NSString *,NSNumber *> *(^)(void))block;

- (SFAnalyticsActivityTracker* _Nullable)logSystemMetricsForActivityNamed:(NSString*)eventName
                                                               withAction:(void (^ _Nullable)(void))action;
- (SFAnalyticsActivityTracker* _Nullable)startLogSystemMetricsForActivityNamed:(NSString *)eventName;
@end

@interface SFAnalytics : NSObject <SFAnalyticsProtocol>

+ (instancetype _Nullable)logger;

+ (NSInteger)fuzzyDaysSinceDate:(NSDate*)date;

// Rounds to the nearest 5 (unless 1 or 2, that rounds to 5 as well)
+ (NSInteger)fuzzyInteger:(NSInteger)num;
+ (NSNumber*)fuzzyNumber:(NSNumber*)num;

+ (void)addOSVersionToEvent:(NSMutableDictionary*)event;
// Help for the subclass to pick a prefered location
+ (NSString *)defaultAnalyticsDatabasePath:(NSString *)basename;
+ (void)removeLegacyDefaultAnalyticsDatabasePath:(NSString *)basename usingDispatchToken:(dispatch_once_t *)onceToken;

+ (NSString *)defaultProtectedAnalyticsDatabasePath:(NSString *)basename uuid:(NSUUID * __nullable)userUuid;
+ (NSString *)defaultProtectedAnalyticsDatabasePath:(NSString *)basename; // uses current user UUID for path

- (void)addMetricsHook:(SFAnalyticsMetricsHook)hook;
- (void)removeMetricsHook:(SFAnalyticsMetricsHook)hook;

- (NSDictionary<NSString*, NSNumber*>*)dailyMetrics;
- (void)dailyCoreAnalyticsMetrics:(NSString *)eventName;

// Log event-based metrics: create an event corresponding to some event in your feature
// and call the appropriate method based on the successfulness of that event
- (void)logSuccessForEventNamed:(NSString*)eventName;
- (void)logSuccessForEventNamed:(NSString*)eventName timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;

- (void)logHardFailureForEventNamed:(NSString*)eventName withAttributes:(NSDictionary* _Nullable)attributes;
- (void)logHardFailureForEventNamed:(NSString*)eventName withAttributes:(NSDictionary* _Nullable)attributes timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;

- (void)logSoftFailureForEventNamed:(NSString*)eventName withAttributes:(NSDictionary* _Nullable)attributes;
- (void)logSoftFailureForEventNamed:(NSString*)eventName withAttributes:(NSDictionary* _Nullable)attributes timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;
- (void)logRockwellFailureForEventNamed:(NSString*)eventName withAttributes:(NSDictionary* _Nullable)attributes;

// or just log an event if it is not failable
- (void)noteEventNamed:(NSString*)eventName;
- (void)noteEventNamed:(NSString*)eventName timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;

- (void)noteLaunchSequence:(SecLaunchSequence *)launchSequence;

- (void)logResultForEvent:(NSString*)eventName
              hardFailure:(bool)hardFailure
                   result:(NSError* _Nullable)eventResultError;
- (void)logResultForEvent:(NSString*)eventName
              hardFailure:(bool)hardFailure
                   result:(NSError* _Nullable)eventResultError
          timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;
- (void)logResultForEvent:(NSString*)eventName
              hardFailure:(bool)hardFailure
                   result:(NSError* _Nullable)eventResultError
           withAttributes:(NSDictionary* _Nullable)attributes;
- (void)logResultForEvent:(NSString*)eventName
              hardFailure:(bool)hardFailure
                   result:(NSError* _Nullable)eventResultError
           withAttributes:(NSDictionary* _Nullable)attributes
          timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;

// Track the state of a named value over time
- (SFAnalyticsSampler* _Nullable)addMetricSamplerForName:(NSString*)samplerName
                                        withTimeInterval:(NSTimeInterval)timeInterval
                                                   block:(NSNumber* (^)(void))block;
- (SFAnalyticsSampler* _Nullable)existingMetricSamplerForName:(NSString*)samplerName;
- (void)removeMetricSamplerForName:(NSString*)samplerName;
// Same idea, but log multiple named values in a single block
- (SFAnalyticsMultiSampler* _Nullable)AddMultiSamplerForName:(NSString*)samplerName
                                            withTimeInterval:(NSTimeInterval)timeInterval
                                                       block:(NSDictionary<NSString*, NSNumber*>* (^)(void))block;
- (SFAnalyticsMultiSampler*)existingMultiSamplerForName:(NSString*)samplerName;
- (void)removeMultiSamplerForName:(NSString*)samplerName;

// Log measurements of arbitrary things
// System metrics measures how much time it takes to complete the action - possibly more in the future. The return value can be ignored if you only need to execute 1 block for your activity
- (SFAnalyticsActivityTracker* _Nullable)logSystemMetricsForActivityNamed:(NSString*)eventName
                                                               withAction:(void (^ _Nullable)(void))action;

// Same as above, but automatically starts the tracker, since you haven't given it any action to perform
- (SFAnalyticsActivityTracker* _Nullable)startLogSystemMetricsForActivityNamed:(NSString *)eventName;

- (void)logMetric:(NSNumber*)metric withName:(NSString*)metricName;

- (void)updateCollectionConfigurationWithData:(NSData * _Nullable)data;
- (void)loadCollectionConfiguration;

// --------------------------------
// Things below are for subclasses

// Override to create a concrete logger instance
@property (readonly, class, nullable) NSString* databasePath;

// Storing dates
- (void)setDateProperty:(NSDate* _Nullable)date forKey:(NSString*)key;
- (NSDate* _Nullable)datePropertyForKey:(NSString*)key;

- (void)incrementIntegerPropertyForKey:(NSString*)key;
- (void)setNumberProperty:(NSNumber* _Nullable)number forKey:(NSString*)key;
- (NSNumber * _Nullable)numberPropertyForKey:(NSString*)key;

- (NSString* _Nullable)metricsAccountID;
- (void)setMetricsAccountID:(NSString* _Nullable)accountID;

// --------------------------------
// Things below are for unit testing

- (void)removeState;    // removes DB object and any samplers
- (void)removeStateAndUnlinkFile:(BOOL)unlinkFile;

@end

@interface SFAnalytics (SFACollection)
+ (NSData * _Nullable)encodeSFACollection:(NSData *_Nonnull)json error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
#endif
#endif
