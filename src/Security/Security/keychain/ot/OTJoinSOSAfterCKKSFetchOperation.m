/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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

#import <utilities/debugging.h>

#import "keychain/ot/OTJoinSOSAfterCKKSFetchOperation.h"
#import "keychain/ot/OTOperationDependencies.h"
#include "keychain/SecureObjectSync/SOSAccount.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/ot/OTWaitOnPriorityViews.h"

@interface OTJoinSOSAfterCKKSFetchOperation ()
@property OTOperationDependencies* operationDependencies;
@property NSOperation* finishedOp;
@end

@implementation OTJoinSOSAfterCKKSFetchOperation
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
{
    if((self = [super init])) {
        _intendedState = intendedState;
        _nextState = errorState;
        _operationDependencies = dependencies;
    }
    return self;
}

- (void)groupStart
{
    if(!self.operationDependencies.sosAdapter.sosEnabled) {
        secnotice("octagon-sos", "SOS not enabled on this platform?");
        self.nextState = self.intendedState;
        return;
    }
    
    secnotice("octagon-sos", "joining SOS");
    
    self.finishedOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishedOp];
    
    OTWaitOnPriorityViews* op = [[OTWaitOnPriorityViews alloc] initWithDependencies:self.operationDependencies];
    
    [op timeout:10*NSEC_PER_SEC];
    
    [self runBeforeGroupFinished:op];
    
    WEAKIFY(self);

    CKKSResultOperation* proceedAfterFetch = [CKKSResultOperation named:@"join-sos-after-fetch"
                                                              withBlock:^{
        STRONGIFY(self);
        [self proceedAfterFetch];
    }];
   
    [proceedAfterFetch addDependency:op];
    [self runBeforeGroupFinished:proceedAfterFetch];
}

- (void)proceedAfterFetch
{
    NSError* restoreError = nil;
    bool restoreResult = [self.operationDependencies.sosAdapter joinAfterRestore:&restoreError];
    
    if (restoreError && restoreError.code == kSOSErrorPrivateKeyAbsent && [restoreError.domain isEqualToString:(id)kSOSErrorDomain]) {
        self.error = restoreError;
        [self runBeforeGroupFinished:self.finishedOp];
        return;
    }
    
    NSError* circleStatusError = nil;
    SOSCCStatus sosCircleStatus = [self.operationDependencies.sosAdapter circleStatus:&circleStatusError];
    if ((circleStatusError && circleStatusError.code == kSOSErrorPrivateKeyAbsent && [circleStatusError.domain isEqualToString:(id)kSOSErrorDomain])
        || sosCircleStatus == kSOSCCError) {
        secnotice("octagon-sos", "Error fetching circle status: %d, error:%@", sosCircleStatus, circleStatusError);
        self.error = circleStatusError;
        [self runBeforeGroupFinished:self.finishedOp];
        return;
    }
    
    if (!restoreResult || restoreError || (sosCircleStatus == kSOSCCRequestPending || sosCircleStatus == kSOSCCNotInCircle)) {
        NSError* resetToOfferingError = nil;
        bool successfulReset = [self.operationDependencies.sosAdapter resetToOffering:&resetToOfferingError];
        
        secnotice("octagon-sos", "SOSCCResetToOffering complete: %d %@", successfulReset, resetToOfferingError);
        if (!successfulReset || resetToOfferingError) {
            self.error = resetToOfferingError;
            [self runBeforeGroupFinished:self.finishedOp];
            return;
        }
    }
    self.nextState = self.intendedState;
    
    [self runBeforeGroupFinished:self.finishedOp];
}

@end

#endif // OCTAGON
