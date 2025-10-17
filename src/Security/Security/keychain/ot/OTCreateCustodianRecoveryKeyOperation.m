/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
#import "keychain/OctagonTrust/OTCustodianRecoveryKey.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/OTCreateCustodianRecoveryKeyOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTFetchCKKSKeysOperation.h"
#import "keychain/ot/ObjCImprovements.h"

#include <Security/SecPasswordGenerate.h>

@interface OTCreateCustodianRecoveryKeyOperation ()
@property OTOperationDependencies* deps;
@property NSUUID* uuid;
@property NSOperation* finishOp;
@end

@implementation OTCreateCustodianRecoveryKeyOperation

- (instancetype)initWithUUID:(NSUUID *_Nullable)uuid dependencies:(OTOperationDependencies*)dependencies
{
    if((self = [super init])) {
        _uuid = uuid ?: [[NSUUID alloc] init];
        _deps = dependencies;
    }
    return self;
}

- (void)groupStart
{
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

    NSString* salt = altDSID;

    WEAKIFY(self);

    OTFetchCKKSKeysOperation* fetchKeysOp = [[OTFetchCKKSKeysOperation alloc] initWithDependencies:self.deps
                                                                                     refetchNeeded:NO];
    [self runBeforeGroupFinished:fetchKeysOp];

    CKKSResultOperation* proceedWithKeys = [CKKSResultOperation named:@"setting-recovery-tlks"
                                                            withBlock:^{
                                                                STRONGIFY(self);
                                                                [self proceedWithKeys:fetchKeysOp.viewKeySets salt:salt];
                                                            }];

    [proceedWithKeys addDependency:fetchKeysOp];
    [self runBeforeGroupFinished:proceedWithKeys];
}

- (void)proceedWithKeys:(NSArray<CKKSKeychainBackedKeySet*>*)viewKeySets salt:(NSString*)salt
{
    WEAKIFY(self);
    
    CFErrorRef cferr = NULL;
    NSString *recoveryString = CFBridgingRelease(SecPasswordGenerate(kSecPasswordTypeiCloudRecovery, &cferr, NULL));
    if (recoveryString == NULL) {
        secerror("octagon: failed to create string: %@", (__bridge id)cferr);
        self.error = CFBridgingRelease(cferr);
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    [self.deps.cuttlefishXPCWrapper createCustodianRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                                   recoveryKey:recoveryString
                                                                          salt:salt
                                                                      ckksKeys:viewKeySets
                                                                          uuid:self.uuid
                                                                          kind:TPPBCustodianRecoveryKey_Kind_RECOVERY_KEY
                                                                         reply:^(NSArray<CKRecord*>* _Nullable keyHierarchyRecords,
                                                                                 TrustedPeersHelperCustodianRecoveryKey *_Nullable crk,
                                                                                 NSError * _Nullable error) {
            STRONGIFY(self);
            [[CKKSAnalytics logger] logResultForEvent:OctagonEventCreateCustodianRecoveryKeyTPH hardFailure:true result:error];
            if(error){
                secerror("octagon: Error create custodian recovery key: %@", error);
                self.error = error;
                [self runBeforeGroupFinished:self.finishOp];
            } else {
                secnotice("octagon", "successfully created custodian recovery key");

                NSError *error = nil;
                NSUUID *uuid = [[NSUUID alloc] initWithUUIDString:crk.uuid];
                if (uuid == nil) {
                    secerror("octagon: failed to parse UUID from TPH: %@", crk.uuid);
                    self.error = [NSError errorWithDomain:OctagonErrorDomain
                                                     code:OctagonErrorBadUUID
                                              description:@"Failed to parse UUID from TPH"];
                    [self runBeforeGroupFinished:self.finishOp];
                    return;
                }
                self.crk = [[OTCustodianRecoveryKey alloc] initWithUUID:uuid
                                                         recoveryString:recoveryString
                                                                  error:&error];
                if (self.crk == nil) {
                    secerror("octagon: failed to create crk: %@", error);
                    self.error = error;
                    [self runBeforeGroupFinished:self.finishOp];
                    return;
                }

                secnotice("octagon-ckks", "Providing createCustodianRecoveryKey() records to %@", self.deps.ckks);
                [self.deps.ckks receiveTLKUploadRecords:keyHierarchyRecords];

                [self runBeforeGroupFinished:self.finishOp];
            }
        }];
}

@end

#endif // OCTAGON
