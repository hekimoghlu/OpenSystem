/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

#import "keychain/ckks/CKKSNewTLKOperation.h"
#import "keychain/ot/OTWaitOnPriorityViews.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTWaitOnPriorityViews ()
@property OTOperationDependencies* operationDependencies;
@end

@implementation OTWaitOnPriorityViews

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
 {
    if((self = [super init])) {
        _operationDependencies = dependencies;
    }
    return self;
}


- (void)groupStart
{
    WEAKIFY(self);
    CKKSResultOperation* proceedAfterFetch = [CKKSResultOperation named:@"proceed-after-fetch"
                                                            withBlock:^{
        STRONGIFY(self);

        [self addNullableSuccessDependency:self.operationDependencies.ckks.zoneChangeFetcher.inflightFetch];
        
        secnotice("octagon-ckks", "Waiting for CKKS Priority view download for %@", self.operationDependencies.ckks);
        [self addSuccessDependency:[self.operationDependencies.ckks rpcProcessIncomingQueue:nil
                                                                       errorOnClassAFailure:false]];
    }];

    [self runBeforeGroupFinished:proceedAfterFetch];
}
@end

#endif // OCTAGON
