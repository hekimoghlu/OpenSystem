/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#import <Foundation/Foundation.h>
#import "keychain/ckks/tests/CKKSMockLockStateProvider.h"
#import "tests/secdmockaks/mockaks.h"

@implementation CKKSMockLockStateProvider
@synthesize aksCurrentlyLocked = _aksCurrentlyLocked;

- (instancetype)initWithCurrentLockStatus:(BOOL)currentlyLocked
{
    if((self = [super init])) {
        _aksCurrentlyLocked = currentlyLocked;
    }
    return self;
}

- (BOOL)queryAKSLocked {
    return self.aksCurrentlyLocked;
}

- (BOOL)aksCurrentlyLocked {
    return _aksCurrentlyLocked;
}

- (void)setAksCurrentlyLocked:(BOOL)aksCurrentlyLocked
{
    if(aksCurrentlyLocked) {
        [SecMockAKS lockClassA];
    } else {
        [SecMockAKS unlockAllClasses];
    }

    _aksCurrentlyLocked = aksCurrentlyLocked;
}

@end

#endif
