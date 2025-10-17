/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#import "keychain/ot/OTSetRecoveryKeyOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTFetchCKKSKeysOperation.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTSetRecoveryKeyOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishOp;
@end

@implementation OTSetRecoveryKeyOperation

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                         recoveryKey:(NSString*)recoveryKey
{
    if((self = [super init])) {
        _deps = dependencies;
        _recoveryKey = recoveryKey;
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

    [self.deps.cuttlefishXPCWrapper setRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                       recoveryKey:self.recoveryKey
                                                              salt:salt
                                                          ckksKeys:viewKeySets
                                                             reply:^(NSArray<CKRecord*>* _Nullable keyHierarchyRecords,
                                                                     NSError * _Nullable setError) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventSetRecoveryKey hardFailure:true result:setError];
        if(setError){
            secerror("octagon: Error setting recovery key: %@", setError);
            self.error = setError;
            [self runBeforeGroupFinished:self.finishOp];
        } else {
            secnotice("octagon", "successfully set recovery key");

            secnotice("octagon-ckks", "Providing setRecoveryKey() records to %@", self.deps.ckks);
            [self.deps.ckks receiveTLKUploadRecords:keyHierarchyRecords];

            [self runBeforeGroupFinished:self.finishOp];
        }
    }];
}

@end

#endif // OCTAGON
