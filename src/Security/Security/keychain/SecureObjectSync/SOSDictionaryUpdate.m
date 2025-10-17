/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

//
//  SOSDIctionaryUpdate.m
//

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonCrypto.h>
#include <dispatch/dispatch.h>
#include <utilities/der_plist.h>
#include <utilities/der_plist_internal.h>
#include <keychain/SecureObjectSync/SOSDictionaryUpdate.h>
#include <utilities/debugging.h>

@interface SOSDictionaryUpdate()
@property dispatch_queue_t queue;
@end

@implementation SOSDictionaryUpdate

@class SOSDictionaryUpdate;

static uint8_t *sosCreateHashFromDict(CFDictionaryRef d) {
    if(!d) {
        return NULL;
    }
    
    CFErrorRef localErr = NULL;
    size_t dersize = der_sizeof_dictionary(d, &localErr);
    
    if(dersize == 0) {
        secnotice("key-interests", "Failed to get size of dictionary - %@", localErr);
        CFReleaseNull(localErr);
        return NULL;
    }
    
    uint8_t derbuf[dersize];
    uint8_t* statptr = der_encode_dictionary(d, &localErr, derbuf, derbuf+dersize);
    if(statptr == NULL) {
        secnotice("key-interests", "Failed to DER encode dictionary - %@", localErr);
        CFReleaseNull(localErr);
        return NULL;
    }
    
    uint8_t *hashbuf = malloc(CC_SHA256_DIGEST_LENGTH);
    CC_SHA256(derbuf, (CC_LONG) dersize, hashbuf);
    return hashbuf;
}

-(id)init
{
    if ((self = [super init])) {
        self->currentHashBuf = NULL;
        _queue = dispatch_queue_create("SOSDictionaryUpdate", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (void)dealloc {
    [self reset];
}

- (void)onqueueFreeHashBuff {
    dispatch_assert_queue(self.queue);

    if(self->currentHashBuf) {
        free(self->currentHashBuf);
        self->currentHashBuf = NULL;
    }
}

- (bool) hasChanged: (CFDictionaryRef) d {
    uint8_t *newHash = sosCreateHashFromDict(d);

    __block bool result = false;

    dispatch_sync(self.queue, ^{
        // If this is the first time or we've reset then currentHashBuf will be null
        // - grab the new hash and indicate things have changed.
        if(self->currentHashBuf == NULL || newHash == NULL) {
            if(self->currentHashBuf == newHash) {
                result = false;
                return;
            }
            [self onqueueFreeHashBuff];
            self->currentHashBuf = newHash;
            result = true;
            return;
        }

        // the hashes of the new vs current bufs aren't equal - the dictionary changed
        // - grab the new hash and indicate things have changed.
        if(memcmp(newHash, self->currentHashBuf, CC_SHA256_DIGEST_LENGTH) != 0) {
            [self onqueueFreeHashBuff];
            self->currentHashBuf = newHash;
            result = true;
            return;
        }

        // if we got here there is no difference between the new and current hashes
        // - report no change.
        if(newHash) {
            free(newHash);
        }
        result = false;
    });

    return result;
}

- (void) reset {
    dispatch_sync(self.queue, ^{
        [self onqueueFreeHashBuff];
    });
}
@end

