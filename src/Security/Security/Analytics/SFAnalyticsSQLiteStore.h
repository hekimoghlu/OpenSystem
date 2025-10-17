/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#import <Security/SFSQLite.h>

#import <Security/SFAnalytics.h>

NS_ASSUME_NONNULL_BEGIN

@interface SFAnalyticsSQLiteStore : SFSQLite

@property (readonly, strong) NSArray* hardFailures;
@property (readonly, strong) NSArray* softFailures;
@property (readonly, strong) NSArray* rockwells;
@property (readonly, strong) NSArray* allEvents;
@property (readonly, strong) NSArray* samples;
@property (readonly, strong) NSString* databaseBasename;
@property (readwrite, strong, nullable) NSDate* uploadDate;
@property (readwrite, strong, nullable) NSString* metricsAccountID;

+ (nullable instancetype)storeWithPath:(NSString*)path schema:(NSString*)schema;

- (BOOL)tryToOpenDatabase;
- (void)incrementSuccessCountForEventType:(NSString*)eventType;
- (void)incrementHardFailureCountForEventType:(NSString*)eventType;
- (void)incrementSoftFailureCountForEventType:(NSString*)eventType;
- (NSInteger)successCountForEventType:(NSString*)eventType;
- (NSInteger)hardFailureCountForEventType:(NSString*)eventType;
- (NSInteger)softFailureCountForEventType:(NSString*)eventType;
- (void)addEventDict:(NSDictionary*)eventDict toTable:(NSString*)table;
- (void)addEventDict:(NSDictionary*)eventDict toTable:(NSString*)table timestampBucket:(SFAnalyticsTimestampBucket)timestampBucket;
- (void)addRockwellDict:(NSString *)eventName
               userinfo:(NSDictionary*)eventDict
                toTable:(NSString*)table
        timestampBucket:(SFAnalyticsTimestampBucket)bucket;
- (void)addSample:(NSNumber*)value forName:(NSString*)name;
- (void)removeAllSamplesForName:(NSString*)name;
- (void)clearAllData;

- (NSDictionary*)summaryCounts;

- (void)streamEventsWithLimit:(NSNumber *_Nullable)limit
                    fromTable:(NSString *)table
                 eventHandler:(bool (^)(NSData * _Nonnull event))eventHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* __OBJC2__ */
