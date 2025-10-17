/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#import "keychain/categories/NSError+UsefulConstructors.h"

#import "keychain/ot/OTVouchWithRecoveryKeyOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTFetchCKKSKeysOperation.h"
#import "keychain/ot/OTStates.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTVouchWithRecoveryKeyOperation ()
@property OTOperationDependencies* deps;

@property NSString* salt;
@property NSString* recoveryKey;

@property NSOperation* finishOp;
@end

@implementation OTVouchWithRecoveryKeyOperation
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                         recoveryKey:(NSString*)recoveryKey
                         saveVoucher:(BOOL)saveVoucher
{
    if((self = [super init])) {
        _deps = dependencies;
        _intendedState = intendedState;
        _nextState = errorState;

        _recoveryKey = recoveryKey;

        _saveVoucher = saveVoucher;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon", "creating voucher using a recovery key");

    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];

    NSString* altDSID = self.deps.activeAccount.altDSID;
    if(altDSID == nil) {
        secnotice("authkit", "No configured altDSID: %@", self.deps.activeAccount);
        self.error = [NSError errorWithDomain:OctagonErrorDomain
                                         code:OctagonErrorNoAppleAccount
                                  description:@"No altDSID configured"];
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    self.salt = altDSID;

    // First, let's preflight the vouch (to receive a policy and view set to use for TLK fetching
    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper preflightVouchWithRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                                      recoveryKey:self.recoveryKey
                                                                             salt:self.salt
                                                                            reply:^(NSString * _Nullable recoveryKeyID,
                                                                                    TPSyncingPolicy* _Nullable peerSyncingPolicy,
                                                                                    NSError * _Nullable error) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventPreflightVouchWithRecoveryKey hardFailure:true result:error];

        if(error || !recoveryKeyID) {
            secerror("octagon: Error preflighting voucher using recovery key: %@", error);
            self.error = error;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }

        secnotice("octagon", "Recovery key ID %@ looks good to go", recoveryKeyID);

        // Tell CKKS to spin up the new views and policy
        // But, do not persist this view set! We'll do that when we actually manage to join
        [self.deps.ckks setCurrentSyncingPolicy:peerSyncingPolicy];

        [self proceedWithRecoveryKeyID:recoveryKeyID];
    }];
}

- (void)proceedWithRecoveryKeyID:(NSString*)recoveryKeyID
{
    WEAKIFY(self);

    [self.deps.cuttlefishXPCWrapper fetchRecoverableTLKSharesWithSpecificUser:self.deps.activeAccount
                                                                       peerID:recoveryKeyID
                                                                      altDSID:self.deps.activeAccount.altDSID
                                                                       flowID:self.deps.flowID
                                                              deviceSessionID:self.deps.deviceSessionID
                                                               canSendMetrics:self.deps.permittedToSendMetrics
                                                                        reply:^(NSArray<CKRecord *> * _Nullable keyHierarchyRecords,
                                                                                NSError * _Nullable error) {
        STRONGIFY(self);

        if(error) {
            secerror("octagon: Error fetching TLKShares to recover: %@", error);
            self.error = error;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }

        NSMutableArray<CKKSTLKShare*>* filteredTLKShares = [NSMutableArray array];
        for(CKRecord* record in keyHierarchyRecords) {
            if([record.recordType isEqual:SecCKRecordTLKShareType]) {
                CKKSTLKShareRecord* tlkShare = [[CKKSTLKShareRecord alloc] initWithCKRecord:record contextID:self.deps.contextID];
                [filteredTLKShares addObject:tlkShare.share];
            }
        }

        [self proceedWithFilteredTLKShares:filteredTLKShares];
    }];
}

- (void)proceedWithFilteredTLKShares:(NSArray<CKKSTLKShare*>*)tlkShares
{
    WEAKIFY(self);

    [self.deps.cuttlefishXPCWrapper vouchWithRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                             recoveryKey:self.recoveryKey
                                                                    salt:self.salt
                                                               tlkShares:tlkShares
                                                                   reply:^(NSData * _Nullable voucher,
                                                                           NSData * _Nullable voucherSig,
                                                                           NSArray<CKKSTLKShare*>* _Nullable newTLKShares,
                                                                           TrustedPeersHelperTLKRecoveryResult* _Nullable tlkRecoveryResults,
                                                                           NSError * _Nullable error) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventVoucherWithRecoveryKey hardFailure:true result:error];
        if(error){
            secerror("octagon: Error preparing voucher using recovery key: %@", error);
            self.error = error;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }


        [[CKKSAnalytics logger] recordRecoveredTLKMetrics:tlkShares
                                       tlkRecoveryResults:tlkRecoveryResults
                                 uniqueTLKsRecoveredEvent:OctagonAnalyticsRKUniqueTLKsRecovered
                                totalSharesRecoveredEvent:OctagonAnalyticsRKTotalTLKSharesRecovered
                           totalRecoverableTLKSharesEvent:OctagonAnalyticsRKTotalTLKShares
                                totalRecoverableTLKsEvent:OctagonAnalyticsRKUniqueTLKsWithSharesCount
                                totalViewsWithSharesEvent:OctagonAnalyticsRKTLKUniqueViewCount];

        self.voucher = voucher;
        self.voucherSig = voucherSig;

        if(self.saveVoucher) {
            secnotice("octagon", "Saving voucher for later use...");
            NSError* saveError = nil;
            [self.deps.stateHolder persistAccountChanges:^OTAccountMetadataClassC * _Nullable(OTAccountMetadataClassC * _Nonnull metadata) {
                metadata.voucher = voucher;
                metadata.voucherSignature = voucherSig;
                [metadata setTLKSharesPairedWithVoucher:newTLKShares];
                return metadata;
            } error:&saveError];
            if(saveError) {
                secnotice("octagon", "unable to save voucher: %@", saveError);
                [self runBeforeGroupFinished:self.finishOp];
                return;
            }
        }

        secnotice("octagon", "Successfully vouched with a recovery key: %@, %@", voucher, voucherSig);
        self.nextState = self.intendedState;
        [self runBeforeGroupFinished:self.finishOp];
    }];
}

@end

#endif // OCTAGON
