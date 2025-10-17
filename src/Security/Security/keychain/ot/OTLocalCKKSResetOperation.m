/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

#import "utilities/debugging.h"

#import "keychain/ot/OTLocalCKKSResetOperation.h"
#import "keychain/ckks/CloudKitCategories.h"
#import "keychain/ckks/CKKSKeychainView.h"
#import "keychain/ckks/CKKSViewManager.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTLocalCKKSResetOperation ()
@property OTOperationDependencies* operationDependencies;

@property NSOperation* finishedOp;
@end

@implementation OTLocalCKKSResetOperation
@synthesize nextState = _nextState;
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
{
    if((self = [super init])) {
        _operationDependencies = dependencies;

        _intendedState = intendedState;
        _nextState = errorState;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon-ckks", "Beginning an 'reset CKKS' operation");

    WEAKIFY(self);

    self.finishedOp = [NSBlockOperation blockOperationWithBlock:^{
        STRONGIFY(self);
        secnotice("octagon-ckks", "Finishing a ckks-local-reset operation with %@", self.error ?: @"no error");
    }];
    [self dependOnBeforeGroupFinished:self.finishedOp];

    [self.operationDependencies.ckks rpcResetLocal:nil reply: ^(NSError* _Nullable resultError) {
        STRONGIFY(self);

        secnotice("octagon-ckks", "Finished ckks-local-reset with %@", self.error ?: @"no error");

        if(resultError == nil) {
            self.nextState = self.intendedState;
        } else {
            self.error = resultError;
        }
        [self runBeforeGroupFinished:self.finishedOp];
    }];
}

@end

#endif // OCTAGON
