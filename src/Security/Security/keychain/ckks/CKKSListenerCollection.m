/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

#import "keychain/ckks/CKKSListenerCollection.h"
#import "keychain/ot/ObjCImprovements.h"

@interface CKKSListenerCollection ()
@property NSString* name;
@property NSMapTable<dispatch_queue_t, id>* listeners;
@end

@implementation CKKSListenerCollection

- (instancetype)initWithName:(NSString*)name
{
    if((self = [super init])) {
        _name = name;
        // Backwards from how we'd like, but it's the best way to have weak pointers to ListenerTypes.
        _listeners = [NSMapTable strongToWeakObjectsMapTable];
    }
    return self;
}

- (NSString*)description
{
    @synchronized(self.listeners) {
        return [NSString stringWithFormat:@"<CKKSListenerCollection(%@): %@>", self.name, [[self.listeners objectEnumerator] allObjects]];
    }
}

- (void)registerListener:(id)listener
{
    @synchronized(self.listeners) {
        bool alreadyRegisteredListener = false;
        NSEnumerator *enumerator = [self.listeners objectEnumerator];
        id value;

        while ((value = [enumerator nextObject])) {
            // actually use pointer comparison
            alreadyRegisteredListener |= (value == listener);
        }

        if(listener && !alreadyRegisteredListener) {
            NSString* queueName = [NSString stringWithFormat: @"%@-%@", self.name, listener];

            dispatch_queue_t objQueue = dispatch_queue_create([queueName UTF8String], DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
            [self.listeners setObject: listener forKey: objQueue];
        }
    }
}

- (void)iterateListeners:(void (^)(id))block
{
    @synchronized(self.listeners) {
        NSEnumerator *enumerator = [self.listeners keyEnumerator];
        dispatch_queue_t dq;

        // Queue up the changes for each listener.
        while ((dq = [enumerator nextObject])) {
            id listener = [self.listeners objectForKey: dq];
            WEAKIFY(listener);

            if(listener) {
                dispatch_async(dq, ^{
                        STRONGIFY(listener);
                        block(listener);
                });
            }
        }
    }
}

@end

#endif // OCTAGON
