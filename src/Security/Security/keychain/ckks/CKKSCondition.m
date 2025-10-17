/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#import "keychain/ckks/CKKSCondition.h"

@interface CKKSCondition ()
@property dispatch_semaphore_t semaphore;
@property CKKSCondition* chain;
@end

@implementation CKKSCondition

-(instancetype)init {
    return [self initToChain:nil];
}

-(instancetype)initToChain:(CKKSCondition*)chain
{
    if((self = [super init])) {
        _semaphore = dispatch_semaphore_create(0);
        _chain = chain;
    }
    return self;
}

-(void)fulfill {
    dispatch_semaphore_signal(self.semaphore);
    [self.chain fulfill];
    self.chain = nil; // break the retain, since that condition is filled
}

-(long)wait:(uint64_t)timeout {
    long result = dispatch_semaphore_wait(self.semaphore, dispatch_time(DISPATCH_TIME_NOW, timeout));

    // If we received a go-ahead from the semaphore, replace the signal
    if(0 == result) {
        dispatch_semaphore_signal(self.semaphore);
    }

    return result;
}

@end

