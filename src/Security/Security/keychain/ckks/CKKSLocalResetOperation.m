/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#import "keychain/categories/NSError+UsefulConstructors.h"
#import "keychain/ckks/CKKSLocalResetOperation.h"
#import "keychain/ckks/CKKSZoneStateEntry.h"
#import "keychain/ckks/CKKSOutgoingQueueEntry.h"
#import "keychain/ckks/CKKSIncomingQueueEntry.h"
#import "keychain/ckks/CKKSCurrentItemPointer.h"
#import "keychain/ckks/CKKSMirrorEntry.h"
#import "keychain/ot/OTDefines.h"

#import <KeychainCircle/SecurityAnalyticsConstants.h>
#import <KeychainCircle/SecurityAnalyticsReporterRTC.h>
#import <KeychainCircle/AAFAnalyticsEvent+Security.h>

@implementation CKKSLocalResetOperation

@synthesize nextState = _nextState;
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(CKKSOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
{
    if(self = [super init]) {
        _deps = dependencies;

        _intendedState = intendedState;
        _nextState = errorState;

        self.name = @"ckks-local-reset";
    }
    return self;
}

- (void)main {
#if TARGET_OS_TV
    [self.deps.personaAdapter prepareThreadForKeychainAPIUseForPersonaIdentifier: nil];
#endif
    [self.deps.databaseProvider dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult {
        [self onqueuePerformLocalReset];
        return CKKSDatabaseTransactionCommit;
    }];
}

- (void)onqueuePerformLocalReset
{
    NSError* localerror = nil;

    AAFAnalyticsEventSecurity *eventS = [[AAFAnalyticsEventSecurity alloc] initWithCKKSMetrics:@{kSecurityRTCFieldNumViews: @(self.deps.views.count)}
                                                                                       altDSID:self.deps.activeAccount.altDSID
                                                                                     eventName:kSecurityRTCEventNameLocalReset
                                                                               testsAreEnabled:SecCKKSTestsEnabled()
                                                                                      category:kSecurityRTCEventCategoryAccountDataAccessRecovery
                                                                                    sendMetric:self.deps.sendMetric];
    for(CKKSKeychainViewState* view in self.deps.views) {
        view.viewKeyHierarchyState = SecCKKSZoneKeyStateResettingLocalData;

        CKKSZoneStateEntry* ckse = [CKKSZoneStateEntry contextID:self.deps.contextID zoneName:view.zoneID.zoneName];
        ckse.ckzonecreated = false;
        ckse.ckzonesubscribed = false; // I'm actually not sure about this: can you be subscribed to a non-existent zone?
        ckse.changeToken = NULL;

        ckse.moreRecordsInCloudKit = NO;
        ckse.lastFetchTime = nil;
        ckse.lastLocalKeychainScanTime = nil;
        ckse.initialSyncFinished = NO;

        [ckse saveToDatabase:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't reset zone status: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSMirrorEntry deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSMirrorEntry: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSOutgoingQueueEntry deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSOutgoingQueueEntry: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSIncomingQueueEntry deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSIncomingQueueEntry: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSKey deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSKey: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSTLKShareRecord deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSTLKShare: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSCurrentKeyPointer deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSCurrentKeyPointer: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSCurrentItemPointer deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSCurrentItemPointer: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        [CKKSDeviceStateEntry deleteAllWithContextID:self.deps.contextID zoneID:view.zoneID error:&localerror];
        if(localerror && self.error == nil) {
            ckkserror("local-reset", view.zoneID, "couldn't delete all CKKSDeviceStateEntry: %@", localerror);
            self.error = localerror;
            localerror = nil;
        }

        if(self.error) {
            break;
        }
    }

    if(!self.error) {
        ckksnotice_global("local-reset", "Successfully deleted all local data for zones: %@", self.deps.views);
        [SecurityAnalyticsReporterRTC sendMetricWithEvent:eventS success:YES error:nil];
        self.nextState = self.intendedState;
    } else {
        [SecurityAnalyticsReporterRTC sendMetricWithEvent:eventS success:NO error:self.error];
    }
}

@end

#endif // OCTAGON

