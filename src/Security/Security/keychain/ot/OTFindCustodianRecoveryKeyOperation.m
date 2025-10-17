/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/OTFindCustodianRecoveryKeyOperation.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTFindCustodianRecoveryKeyOperation ()
@property OTOperationDependencies* deps;
@property NSUUID* uuid;
@property NSOperation* finishOp;
@end

@implementation OTFindCustodianRecoveryKeyOperation

- (instancetype)initWithUUID:(NSUUID *)uuid dependencies:(OTOperationDependencies*)dependencies
{
    if((self = [super init])) {
        _uuid = uuid;
        _deps = dependencies;
    }
    return self;
}

- (void)groupStart
{
    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];
    
    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper findCustodianRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                                        uuid:self.uuid
                                                                       reply:^(TrustedPeersHelperCustodianRecoveryKey *_Nullable crk, NSError * _Nullable error) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventCheckCustodianRecoveryKeyTPH hardFailure:true result:error];
        if(error) {
            secerror("octagon: Error finding custodian recovery key: %@", error);
            self.error = error;
            [self runBeforeGroupFinished:self.finishOp];
        } else {
            secnotice("octagon", "successfully found custodian recovery key");
            [self runBeforeGroupFinished:self.finishOp];
            self.crk = crk;
        }
    }];
}

@end

#endif // OCTAGON
