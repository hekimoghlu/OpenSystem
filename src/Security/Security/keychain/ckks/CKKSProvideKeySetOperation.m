/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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

#import "CKKSProvideKeySetOperation.h"

@interface CKKSProvideKeySetOperation ()
@property (nullable) NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>* keysets;
@property dispatch_queue_t queue;

@property (nullable) NSOperation* startDependency;
@end

@implementation CKKSProvideKeySetOperation
@synthesize keysets = _keysets;
@synthesize intendedZoneIDs = _intendedZoneIDs;

- (instancetype)initWithIntendedZoneIDs:(NSSet<CKRecordZoneID*>*)intendedZoneIDs
{
    if((self = [super init])) {
        _intendedZoneIDs = intendedZoneIDs;
        _keysets = nil;
        _startDependency = [NSBlockOperation blockOperationWithBlock:^{}];
        _startDependency.name = @"key-set-provided";

        _queue = dispatch_queue_create("key-set-queue", DISPATCH_QUEUE_SERIAL);

        [self addDependency:_startDependency];
    }
    return self;
}

- (void)provideKeySets:(NSDictionary<CKRecordZoneID*, CKKSCurrentKeySet*>*)keysets
{
    // Ensure that only one keyset groupt is provided through each operation
    dispatch_sync(self.queue, ^{
        if(!self.keysets) {
            self.keysets = keysets;
            if(self.startDependency) {
                // Create a new queue here, just to be safe in case someone is waiting
                NSOperationQueue* queue = [[NSOperationQueue alloc] init];
                [queue addOperation:self.startDependency];
                self.startDependency = nil;
            }
        }
    });
}

@end

#endif // OCTAGON
