/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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

#import "keychain/ot/OTVouchWithRerollOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTFetchCKKSKeysOperation.h"
#import "keychain/ot/OTStates.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTVouchWithRerollOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishOp;

@property NSString* peerID;
@property NSString* oldPeerID;
@property TPSyncingPolicy* syncingPolicy;
@end

@implementation OTVouchWithRerollOperation
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                         saveVoucher:(BOOL)saveVoucher
{
    if((self = [super init])) {
        _deps = dependencies;
        _intendedState = intendedState;
        _nextState = errorState;
        _saveVoucher = saveVoucher;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon", "creating voucher for reroll");

    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];

    NSError* error = nil;
    OTAccountMetadataClassC* accountState = [self.deps.stateHolder loadOrCreateAccountMetadata:&error];
    if (error != nil) {
        secerror("octagon: Error loading account metadata: %@", error);
        self.error = error;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    self.peerID = accountState.peerID;
    self.oldPeerID = accountState.oldPeerID;
    self.syncingPolicy = [accountState getTPSyncingPolicy];

    WEAKIFY(self);

    [self.deps.cuttlefishXPCWrapper fetchRecoverableTLKSharesWithSpecificUser:self.deps.activeAccount
                                                                       peerID:self.peerID
                                                                      altDSID:self.deps.activeAccount.altDSID
                                                                       flowID:self.deps.flowID
                                                              deviceSessionID:self.deps.deviceSessionID
                                                               canSendMetrics:self.deps.permittedToSendMetrics
                                                                        reply:^(NSArray<CKRecord *> * _Nullable keyHierarchyRecords,
                                                                                NSError * _Nullable error) {
        STRONGIFY(self);

        if(error) {
            secerror("octagon: Error fetching TLKShares to recover: %@", error);
            // recovering these is best-effort, so fall through.
            // Note: if there are any TLKShares to our own peer ID, then this device should already have the TLKs and not need this fetch.
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

    [self.deps.cuttlefishXPCWrapper vouchWithRerollWithSpecificUser:self.deps.activeAccount
                                                          oldPeerID:self.oldPeerID
                                                               tlkShares:tlkShares
                                                                   reply:^(NSData * _Nullable voucher,
                                                                           NSData * _Nullable voucherSig,
                                                                           NSArray<CKKSTLKShare*>* _Nullable newTLKShares,
                                                                           TrustedPeersHelperTLKRecoveryResult* _Nullable tlkRecoveryResults,
                                                                           NSError * _Nullable error) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventVoucherWithReroll hardFailure:true result:error];
        if(error){
            secerror("octagon: Error preparing voucher using reroll: %@", error);
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

        secnotice("octagon", "Successfully vouched with a reroll: %@, %@", voucher, voucherSig);
        self.nextState = self.intendedState;
        [self runBeforeGroupFinished:self.finishOp];
    }];
}

@end

#endif // OCTAGON
