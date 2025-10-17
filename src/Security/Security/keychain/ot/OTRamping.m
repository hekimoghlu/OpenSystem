/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
#if OCTAGON

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <CloudKit/CloudKit.h>
#import <CloudKit/CKContainer_Private.h>
#import <utilities/debugging.h>
#import "OTRamping.h"
#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSNearFutureScheduler.h"
#import "keychain/ckks/CKKSAnalytics.h"
#import "keychain/ot/OTDefines.h"
#import "keychain/ot/OTConstants.h"
#import "keychain/ckks/CKKS.h"

static NSString* kFeatureAllowedKey =       @"FeatureAllowed";
static NSString* kFeaturePromotedKey =      @"FeaturePromoted";
static NSString* kFeatureVisibleKey =       @"FeatureVisible";
static NSString* kRetryAfterKey =           @"RetryAfter";
static NSString* kRampPriorityKey =         @"RampPriority";

#define kCKRampManagerDefaultRetryTimeInSeconds 86400

#if OCTAGON
@interface OTRamp (lockstateTracker) <CKKSLockStateNotification>
@end
#endif

@interface OTRamp ()
@property (nonatomic, strong) CKContainer   *container;
@property (nonatomic, strong) CKDatabase    *database;

@property (nonatomic, strong) CKRecordZone      *zone;
@property (nonatomic, strong) CKRecordZoneID    *zoneID;

@property (nonatomic, strong) NSString      *recordName;
@property (nonatomic, strong) NSString      *localSettingName;

@property (nonatomic, strong) CKKSAccountStateTracker *accountTracker;
@property (nonatomic, strong) CKKSLockStateTracker      *lockStateTracker;
@property (nonatomic, strong) CKKSReachabilityTracker   *reachabilityTracker;

@property CKKSAccountStatus accountStatus;

@property (readonly) Class<CKKSFetchRecordsOperation> fetchRecordRecordsOperationClass;

@property (atomic, strong) NSDate *lastFetch;
@property (atomic) NSTimeInterval retryAfter;
@property (atomic) BOOL cachedFeatureAllowed;

@end

@implementation OTRamp

-(instancetype) initWithRecordName:(NSString *) recordName
                  localSettingName:(NSString*) localSettingName
                         container:(CKContainer*) container
                          database:(CKDatabase*) database
                            zoneID:(CKRecordZoneID*) zoneID
                    accountTracker:(CKKSAccountStateTracker*) accountTracker
                  lockStateTracker:(CKKSLockStateTracker*) lockStateTracker
               reachabilityTracker:(CKKSReachabilityTracker*) reachabilityTracker
fetchRecordRecordsOperationClass:(Class<CKKSFetchRecordsOperation>) fetchRecordRecordsOperationClass

{
    if ((self = [super init])) {
        _container = container;
        _recordName = [recordName copy];
        _localSettingName = [localSettingName copy];
        _database = database;
        _zoneID = zoneID;
        _accountTracker = accountTracker;
        _lockStateTracker = lockStateTracker;
        _reachabilityTracker = reachabilityTracker;
        _fetchRecordRecordsOperationClass = fetchRecordRecordsOperationClass;
        _lastFetch = [NSDate distantPast];
        _retryAfter = kCKRampManagerDefaultRetryTimeInSeconds;
        _cachedFeatureAllowed = NO;
    }
    return self;
}

-(void)fetchRampRecordWithReply:(void (^)(BOOL featureAllowed, BOOL featurePromoted, BOOL featureVisible, NSInteger retryAfter, NSError *rampStateFetchError))recordRampStateFetchCompletionBlock
{
    CKOperationConfiguration *opConfig = [[CKOperationConfiguration alloc] init];
    opConfig.allowsCellularAccess = YES;
    opConfig.isCloudKitSupportOperation = YES;

    CKRecordID *recordID = [[CKRecordID alloc] initWithRecordName:_recordName zoneID:_zoneID];
    CKFetchRecordsOperation *operation = [[[self.fetchRecordRecordsOperationClass class] alloc] initWithRecordIDs:@[ recordID]];

    operation.desiredKeys = @[kFeatureAllowedKey, kFeaturePromotedKey, kFeatureVisibleKey, kRetryAfterKey];
    
    operation.configuration = opConfig;
    operation.fetchRecordsCompletionBlock = ^(NSDictionary<CKRecordID *,CKRecord *> * _Nullable recordsByRecordID, NSError * _Nullable operationError) {
        BOOL featureAllowed = NO;
        BOOL featurePromoted = NO;
        BOOL featureVisible = NO;
        NSInteger retryAfter = kCKRampManagerDefaultRetryTimeInSeconds;

        secnotice("octagon", "Fetch operation records %@ fetchError %@", recordsByRecordID, operationError);
        // There should only be only one record.
        CKRecord *rampRecord = recordsByRecordID[recordID];

        if (rampRecord) {
            featureAllowed = [rampRecord[kFeatureAllowedKey] boolValue];
            featurePromoted = [rampRecord[kFeaturePromotedKey] boolValue];
            featureVisible = [rampRecord[kFeatureVisibleKey] boolValue];
            retryAfter = [rampRecord[kRetryAfterKey] integerValue];

            secnotice("octagon", "Fetch ramp state - featureAllowed %@, featurePromoted: %@, featureVisible: %@, retryAfter: %ld", (featureAllowed ? @YES : @NO), (featurePromoted ? @YES : @NO), (featureVisible ? @YES : @NO), (long)retryAfter);
        } else {
            secerror("octagon: Couldn't find CKRecord for ramp. Defaulting to not ramped in");
            operationError = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorRecordNotFound userInfo:@{NSLocalizedDescriptionKey: @" Couldn't find CKRecord for ramp. Defaulting to not ramped in"}];
        }
        recordRampStateFetchCompletionBlock(featureAllowed, featurePromoted, featureVisible, retryAfter, operationError);
    };

    [self.database addOperation: operation];
    secnotice("octagon", "Attempting to fetch ramp state from CloudKit");
}

-(BOOL) checkRampStateWithError:(NSError**)error
{
    __block BOOL isFeatureEnabled = NO;
    __block NSError* localError = nil;
    __block NSInteger localRetryAfter = 0;

    //defaults write to for whether or not a ramp record returns "enabled or disabled"
    CFBooleanRef enabled = (CFBooleanRef)CFPreferencesCopyValue((__bridge CFStringRef)self.localSettingName,
                                                                CFSTR("com.apple.security"),
                                                                kCFPreferencesCurrentUser, kCFPreferencesAnyHost);

    secnotice("octagon", "%@ Defaults availability: SecCKKSTestsEnabled[%s] DefaultsPointer[%s] DefaultsValue[%s]", (__bridge CFStringRef)self.localSettingName,
              SecCKKSTestsEnabled() ? "True": "False", (enabled != NULL) ? "True": "False",
              (enabled && (CFGetTypeID(enabled) == CFBooleanGetTypeID()) && (enabled == kCFBooleanTrue)) ? "True": "False");
    
    if(!SecCKKSTestsEnabled() && enabled && CFGetTypeID(enabled) == CFBooleanGetTypeID()){
        BOOL localConfigEnable = (enabled == kCFBooleanTrue);
        secnotice("octagon", "feature is %@: %@ (local config)", localConfigEnable ? @"enabled" : @"disabled", self.recordName);
        CFReleaseNull(enabled);
        return localConfigEnable;
    }
    CFReleaseNull(enabled);
    
    NSDate* now = [[NSDate alloc] init];

    if([now timeIntervalSinceDate: self.lastFetch] < self.retryAfter) {
        return self.cachedFeatureAllowed;
    }

    if(self.lockStateTracker.isLocked){
        secnotice("octagon","device is locked! can't check ramp state");
        localError = [NSError errorWithDomain:(__bridge NSString*)kSecErrorDomain
                                         code:errSecInteractionNotAllowed
                                     userInfo:@{NSLocalizedDescriptionKey: @"device is locked"}];
        if(error){
            *error = localError;
        }
        return NO;
    }

    // Wait until the account tracker has had a chance to figure out the state
    [self.accountTracker.ckAccountInfoInitialized wait:5*NSEC_PER_SEC];
    if(self.accountTracker.currentCKAccountInfo.accountStatus != CKAccountStatusAvailable){
        secnotice("octagon","not signed in! can't check ramp state");
        localError = [NSError errorWithDomain:OctagonErrorDomain
                                         code:OctagonErrorNotSignedIn
                                     userInfo:@{NSLocalizedDescriptionKey: @"not signed in"}];
        if(error){
            *error = localError;
        }
        return NO;
    }
    if(!self.reachabilityTracker.currentReachability){
        secnotice("octagon","no network! can't check ramp state");
        localError = [NSError errorWithDomain:OctagonErrorDomain
                                         code:OctagonErrorNoNetwork
                                     userInfo:@{NSLocalizedDescriptionKey: @"no network"}];
        if(error){
            *error = localError;
        }
        return NO;
    }

    CKKSAnalytics* logger = [CKKSAnalytics logger];
    SFAnalyticsActivityTracker *tracker = [logger logSystemMetricsForActivityNamed:CKKSActivityOTFetchRampState withAction:nil];

    dispatch_semaphore_t sema = dispatch_semaphore_create(0);

    [tracker start];

    [self fetchRampRecordWithReply:^(BOOL featureAllowed, BOOL featurePromoted, BOOL featureVisible, NSInteger retryAfter, NSError *rampStateFetchError) {
        secnotice("octagon", "fetch ramp records returned with featureAllowed: %d,\n featurePromoted: %d,\n featureVisible: %d,\n", featureAllowed, featurePromoted, featureVisible);

        isFeatureEnabled = featureAllowed;
        localRetryAfter = retryAfter;
        if(rampStateFetchError){
            localError = rampStateFetchError;
        }
        dispatch_semaphore_signal(sema);
    }];

    int64_t timeout = (int64_t)(SecCKKSTestsEnabled() ? 2*NSEC_PER_SEC : NSEC_PER_SEC * 65);
    if(dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, timeout)) != 0) {
        secnotice("octagon", "timed out waiting for response from CloudKit\n");
        localError = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorCKTimeOut userInfo:@{NSLocalizedDescriptionKey: @"timed out waiting for response from CloudKit"}];

        [logger logUnrecoverableError:localError forEvent:OctagonEventRamp withAttributes:@{
                                                                                            OctagonEventAttributeFailureReason : @"cloud kit timed out"}
         ];
    }

    [tracker stop];

    if(localRetryAfter > 0){
        secnotice("octagon", "cloud kit asked security to retry: %lu", (unsigned long)localRetryAfter);
        self.retryAfter = localRetryAfter;
    }

    if(localError){
        secerror("octagon: had an error fetching ramp state: %@", localError);
        [logger logUnrecoverableError:localError forEvent:OctagonEventRamp withAttributes:@{
                                                                                            OctagonEventAttributeFailureReason : @"fetching ramp state"}
         ];
        if(error){
            *error = localError;
        }
    }
    if(isFeatureEnabled){
        [logger logSuccessForEventNamed:OctagonEventRamp];
    }

    self.lastFetch = now;
    self.cachedFeatureAllowed = isFeatureEnabled;
    return isFeatureEnabled;
}

@end
#endif


