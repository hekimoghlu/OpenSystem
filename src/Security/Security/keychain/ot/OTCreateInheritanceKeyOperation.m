/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

#import "keychain/OctagonTrust/OTInheritanceKey.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/OTCreateInheritanceKeyOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTFetchCKKSKeysOperation.h"
#import "keychain/ot/ObjCImprovements.h"

#include <Security/SecPasswordGenerate.h>

@interface OTCreateInheritanceKeyOperation ()
@property OTOperationDependencies* deps;
@property NSUUID* uuid;
@property NSOperation* finishOp;
@end

@implementation OTCreateInheritanceKeyOperation

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
    
    NSString* salt = @"";

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
    
    NSError *error = nil;
    self.ik = [[OTInheritanceKey alloc] initWithUUID:self.uuid error:&error];
    if (self.ik == nil) {
        secerror("octagon: failed to create inheritance key: %@", error);
        self.error = error;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    NSString *str = [self.ik.recoveryKeyData base64EncodedStringWithOptions:0];

    [self.deps.cuttlefishXPCWrapper createCustodianRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                                   recoveryKey:str
                                                                          salt:salt
                                                                      ckksKeys:viewKeySets
                                                                          uuid:self.uuid
                                                                          kind:TPPBCustodianRecoveryKey_Kind_INHERITANCE_KEY
                                                                         reply:^(NSArray<CKRecord*>* _Nullable keyHierarchyRecords,
                                                                                 TrustedPeersHelperCustodianRecoveryKey *_Nullable crk,
                                                                                 NSError * _Nullable error) {
            STRONGIFY(self);
            [[CKKSAnalytics logger] logResultForEvent:OctagonEventCreateCustodianRecoveryKeyTPH hardFailure:true result:error];
            if(error){
                secerror("octagon: Error create inheritance key: %@", error);
                self.error = error;
                [self runBeforeGroupFinished:self.finishOp];
            } else {
                secnotice("octagon", "successfully created inheritance key");

                secnotice("octagon-ckks", "Providing createCustodianRecoveryKey() records to %@", self.deps.ckks);
                [self.deps.ckks receiveTLKUploadRecords:keyHierarchyRecords];
                [self runBeforeGroupFinished:self.finishOp];
            }
        }];
}

@end

#endif // OCTAGON
