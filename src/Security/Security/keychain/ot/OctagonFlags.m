/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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

#import "keychain/ot/OctagonFlags.h"
#import "keychain/ot/OTStates.h"
#import "keychain/ckks/CKKSCondition.h"

@interface OctagonFlags ()
@property dispatch_queue_t queue;
@property NSMutableSet<OctagonFlag*>* flags;
@property (readonly) NSSet* allowableFlags;
@end

@implementation OctagonFlags

- (instancetype)initWithQueue:(dispatch_queue_t)queue
                       flags:(NSSet<OctagonFlag*>*)possibleFlags
{
    if((self = [super init])) {
        _queue = queue;
        _flags = [NSMutableSet set];
        _flagConditions = [[NSMutableDictionary alloc] init];
        _allowableFlags = possibleFlags;

        for(OctagonFlag* flag in possibleFlags) {
            self.flagConditions[flag] = [[CKKSCondition alloc] init];
        }
    }
    return self;
}

- (NSString*)description
{
    return [NSString stringWithFormat:@"<OctagonFlags: %@>", [self contentsAsString]];
}

- (NSString*)contentsAsString
{
    if(self.flags.count == 0) {
        return @"none";
    }
    return [[self.flags allObjects] componentsJoinedByString:@","];
}

- (NSArray<NSString*>*)dumpFlags
{
    return [self.flags allObjects];
}

- (BOOL)_onqueueContains:(nonnull OctagonFlag *)flag {
    dispatch_assert_queue(self.queue);
    return [self.flags containsObject:flag];
}

- (void)_onqueueSetFlag:(nonnull OctagonFlag *)flag {
    dispatch_assert_queue(self.queue);
    [self.flags addObject:flag];
}

- (CKKSCondition*)conditionForFlag:(OctagonFlag*)flag {
    return self.flagConditions[flag];
}

- (CKKSCondition* _Nullable)conditionForFlagIfPresent:(OctagonFlag*)flag {
    __block CKKSCondition* ret = nil;

    dispatch_sync(self.queue, ^{
        if([self.flags containsObject:flag]) {
            ret = self.flagConditions[flag];
        }
    });

    return ret;
}

- (void)setFlag:(nonnull OctagonFlag *)flag {
    dispatch_sync(self.queue, ^{
        [self _onqueueSetFlag:flag];
    });
}

- (void)_onqueueRemoveFlag:(nonnull OctagonFlag *)flag {
    dispatch_assert_queue(self.queue);

    NSAssert([self.allowableFlags containsObject:flag], @"state machine tried to handle unknown flag %@", flag);

    [self.flags removeObject:flag];
    [self.flagConditions[flag] fulfill];
    self.flagConditions[flag] = [[CKKSCondition alloc]init];
}

@end

#endif
