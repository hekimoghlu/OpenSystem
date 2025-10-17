/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

@class SECSFAEventRule;
@class SECSFAVersion;

NS_ASSUME_NONNULL_BEGIN

@interface SFAnalyticsMatchingRule: NSObject
@property NSString *eventName;
@property (readonly) SECSFAEventRule *rule;
+ (NSString *)armKeyForEventName:(NSString *)eventName;
@end

@protocol SFAnalyticsCollectionAction <NSObject>
- (BOOL)shouldRatelimit:(SFAnalytics*)logger rule:(SFAnalyticsMatchingRule*)rule;
- (void)autoBugCaptureWithType:(NSString *)type  subType:(NSString *)subType domain:(NSString *)domain;
- (void)tapToRadar:(NSString*)alert
       description:(NSString*)description
             radar:(NSString*)radar
     componentName:(NSString*)componentName
  componentVersion:(NSString*)componentVersion
       componentID:(NSString*)componentID
        attributes:(NSDictionary* _Nullable)attributes;

@end

typedef NSMutableDictionary<NSString*, NSMutableSet<SFAnalyticsMatchingRule*>*> SFAMatchingRules;

@interface SecSFAParsedCollection: NSObject
@property SFAMatchingRules *matchingRules;
@property NSMutableDictionary<NSString*,NSNumber*>* allowedEvents;
@property (readwrite) BOOL excludedVersion;
@end

@interface SFAnalyticsCollection : NSObject

- (instancetype)init;
- (instancetype)initWithActionInterface:(id<SFAnalyticsCollectionAction>)action
                                product:(NSString *)product
                                  build:(NSString *)build;

- (void)loadCollection:(SFAnalytics *)logger;
- (void)storeCollection:(NSData * _Nullable)data logger:(SFAnalytics *_Nullable)logger;
- (void)stopMetricCollection;

+ (SECSFAVersion *_Nullable)parseVersion:(NSString *)build platform:(NSString *)platform;
+ (BOOL)isVersionSameOrNewer:(SECSFAVersion *)v1 than:(SECSFAVersion *)v2;


- (SFAnalyticsMetricsHookActions)match:(NSString*)eventName
                            eventClass:(SFAnalyticsEventClass)eventClass
                            attributes:(NSDictionary*)attributes
                                bucket:(SFAnalyticsTimestampBucket)timestampBucket
                                logger:(SFAnalytics *)logger;

- (SecSFAParsedCollection *_Nullable)parseCollection:(NSData *)data
                                        logger:(SFAnalytics *)logger;

// only for testing
@property (readonly) BOOL excludedVersion;
@property (readonly) SFAMatchingRules *matchingRules;
@property NSString *processName;
- (void)drainSetupQueue;

@end

NS_ASSUME_NONNULL_END
