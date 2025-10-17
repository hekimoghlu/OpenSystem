/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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

#import "keychain/ot/OTCheckHealthOperation.h"
#import "keychain/ot/OTOperationDependencies.h"
#import "keychain/ot/OTStates.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import <Security/SecInternalReleasePriv.h>
#import <CloudServices/SecureBackup.h>

#if !TARGET_OS_SIMULATOR
#import <MobileKeyBag/MobileKeyBag.h>
#endif

#if TARGET_OS_MAC && !TARGET_OS_SIMULATOR
#include <unistd.h>
#endif

@interface OTCheckHealthOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishOp;
@property BOOL requiresEscrowCheck;
@end

@implementation OTCheckHealthOperation
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                          deviceInfo:(nonnull OTDeviceInformation *)deviceInfo
                skipRateLimitedCheck:(BOOL)skipRateLimitedCheck
             reportRateLimitingError:(BOOL)reportRateLimitingError
                              repair:(BOOL)repair
{
    if((self = [super init])) {
        _deps = dependencies;
        _intendedState = intendedState;
        _nextState = errorState;
        _results = nil;
        _skipRateLimitingCheck = skipRateLimitedCheck;
        _reportRateLimitingError = reportRateLimitingError;
        _repair = repair;
    }
    return self;
}

- (BOOL) checkIfPasscodeIsSetForDevice
{
    BOOL passcodeIsSet = NO;
#if TARGET_OS_MAC && !TARGET_OS_SIMULATOR
    aks_device_state_s deviceState;
    kern_return_t retCode = aks_get_device_state(session_keybag_handle, &deviceState);
    if (kAKSReturnSuccess == retCode){
        passcodeIsSet = (deviceState.lock_state != keybag_lock_disabled ? YES : NO);
    } else {
        secerror("octagon-health: aks_get_device_state failed with: %d", retCode);
    }
    secnotice("octagon-health", "checkIfPasscodeIsSetForDevice is %{BOOL}d", passcodeIsSet);
#endif
    return passcodeIsSet;
}

- (void)groupStart
{
    secnotice("octagon-health", "Beginning cuttlefish health checkup");

    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];

    if(self.skipRateLimitingCheck == NO) {
        secnotice("octagon-health", "running rate limiting checks!");
        NSDate* lastUpdate = nil;
        NSError* accountLoadError = nil;
        self.error = nil;

        lastUpdate = [self.deps.stateHolder lastHealthCheckupDate:&accountLoadError];

        if([self.deps.lockStateTracker isLockedError:accountLoadError]) {
            secnotice("octagon-health", "device is locked, not performing cuttlefish check");
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }
        secnotice("octagon-health", "last health check timestamp: %@", lastUpdate);

        // Only query cuttlefish for trust status every 3 days (1 day for internal installs)
        NSDateComponents* offset = [[NSDateComponents alloc] init];
        if(SecIsInternalRelease()) {
            [offset setHour:-23];
        } else {
            [offset setHour:-3*24];
        }
        NSDate *now = [NSDate date];
        NSDate* deadline = [[NSCalendar currentCalendar] dateByAddingComponents:offset toDate:now options:0];

        if(lastUpdate == nil || [lastUpdate compare: deadline] == NSOrderedAscending) {
            secnotice("octagon-health", "Not rate-limiting: last updated %@ vs %@", lastUpdate, deadline);
        } else {
            secnotice("octagon-health", "Last update is within 3 days (%@); rate-limiting this operation", lastUpdate);
            NSString *description = [NSString stringWithFormat:@"Rate-limited the OTCheckHealthOperation:%@", lastUpdate];
            NSError *rateLimitedError =  [NSError errorWithDomain:@"securityd"
                                                             code:errSecInternalError
                                                         userInfo:@{NSLocalizedDescriptionKey: description}];
            secnotice("octagon-health", "rate limited! %@", rateLimitedError);
            if (self.reportRateLimitingError) {
                self.error = rateLimitedError;
            } else {
                self.nextState = self.intendedState; //not setting the error on the results op as I don't want a CFU posted.
            }
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }
        NSError* persistedError = nil;
        BOOL persisted = [self.deps.stateHolder persistLastHealthCheck:now error:&persistedError];

        if([self.deps.lockStateTracker isLockedError:persistedError]) {
            secnotice("octagon-health", "device is locked, not performing cuttlefish check");
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }
        if(persisted == NO || persistedError) {
            secerror("octagon-health: failed to persist last health check value:%@", persistedError);
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }
    } else {
        secnotice("octagon-health", "NOT running rate limiting checks!");
    }
    WEAKIFY(self);

    [self.deps.cuttlefishXPCWrapper requestHealthCheckWithSpecificUser:self.deps.activeAccount
                                                   requiresEscrowCheck:[self checkIfPasscodeIsSetForDevice]
                                                                repair:self.repair
#if TARGET_OS_TV
                                                      knownFederations:[NSArray array]
#else
                                                      knownFederations:[SecureBackup knownICDPFederations:NULL]
#endif
                                                                flowID:self.deps.flowID
                                                       deviceSessionID:self.deps.deviceSessionID 
                                                                 reply:^(TrustedPeersHelperHealthCheckResult* result, NSError *error) {
            STRONGIFY(self);
            if(error) {
                secerror("octagon-health: error: %@", error);
                self.error = error;

                [self runBeforeGroupFinished:self.finishOp];
                return;
            } else {
                secnotice("octagon-health", "cuttlefish came back with these suggestions: %@", result);
                [self handleRepairSuggestions:result];
            }
        }];
}

- (void)handleRepairSuggestions:(TrustedPeersHelperHealthCheckResult*)results
{
    self.results = results;

    if (self.results.resetOctagon) {
        secnotice("octagon-health", "Resetting Octagon as per Cuttlefish request");
        self.nextState = OctagonStateHealthCheckReset;
    } else if(self.results.leaveTrust) {
        secnotice("octagon-health", "Leaving clique as per Cuttlefish request");
        self.nextState = OctagonStateHealthCheckLeaveClique;
    } else {
        self.nextState = self.intendedState;
    }

    [self runBeforeGroupFinished:self.finishOp];
}

@end

#endif // OCTAGON
