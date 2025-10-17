/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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


#import <Foundation/Foundation.h>
#import <Foundation/NSXPCConnection_Private.h>
#import <Security/SecItemPriv.h>
#import <Security/Security.h>
#import <xpc/xpc.h>

#include "utilities/SecCFWrappers.h"
#include "utilities/SecInternalReleasePriv.h"
#import "utilities/debugging.h"

#import "keychain/otctl/EscrowRequestCLI.h"

@implementation EscrowRequestCLI

- (instancetype)initWithEscrowRequest:(SecEscrowRequest*)escrowRequest
{
    if((self = [super init])) {
        _escrowRequest = escrowRequest;
    }

    return self;
}

- (int)trigger
{
    NSError* error = nil;
    [self.escrowRequest triggerEscrowUpdate:@"cli" error:&error];

    if(error) {
        printf("Errored: %s", [[error description] UTF8String]);
        return 1;

    } else {
        printf("Complete.\n");
    }
    return 0;
}

- (int)status
{
    NSError* error = nil;
    NSDictionary* statuses = [self.escrowRequest fetchStatuses:&error];

    if(error) {
        printf("Errored: %s\n", [[error description] UTF8String]);
        return 1;
    }

    if(statuses.count == 0) {
        printf("No requests are waiting for a passcode.\n");
        return 0;
    }

    for(NSString* uuid in statuses.allKeys) {
        printf("Request %s: %s\n", [uuid UTF8String], [[statuses[uuid] description] UTF8String]);
    }

    return 0;
}

- (int)reset
{
    NSError* error = nil;
    [self.escrowRequest resetAllRequests:&error];

    if(error) {
        printf("Errored: %s\n", [[error description] UTF8String]);
        return 1;
    }

    printf("Complete.\n");
    return 0;
}

- (int)storePrerecordsInEscrow
{
    NSError* error = nil;
    uint64_t recordsWritten = [self.escrowRequest storePrerecordsInEscrow:&error];

    if(error) {
        printf("Errored: %s\n", [[error description] UTF8String]);
        return 1;
    }

    printf("Complete: %d records written.\n", (int)recordsWritten);
    return 0;
}

@end
