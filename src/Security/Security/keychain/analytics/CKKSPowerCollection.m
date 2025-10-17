/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#import "CKKSPowerCollection.h"
#import "CKKSOutgoingQueueEntry.h"
#import "SecPLWrappers.h"

#if OCTAGON

CKKSPowerEvent* const kCKKSPowerEventOutgoingQueue = (CKKSPowerEvent*)@"processOutgoingQueue";
CKKSPowerEvent* const kCKKSPowerEventIncommingQueue = (CKKSPowerEvent*)@"processIncomingQueue";
CKKSPowerEvent* const kCKKSPowerEventTLKShareProcessing = (CKKSPowerEvent*)@"TLKShareProcessing";
CKKSPowerEvent* const kCKKSPowerEventScanLocalItems = (CKKSPowerEvent*)@"scanLocalItems";
CKKSPowerEvent* const kCKKSPowerEventFetchAllChanges = (CKKSPowerEvent*)@"fetchAllChanges";
CKKSPowerEvent* const kCKKSPowerEventReencryptOutgoing = (CKKSPowerEvent *)@"reencryptOutgoing";

OTPowerEvent* const kOTPowerEventRestore = (OTPowerEvent *)@"restoreBottledPeer";
OTPowerEvent* const kOTPowerEventEnroll = (OTPowerEvent *)@"enrollBottledPeer";


@interface CKKSPowerCollection ()
@property (strong) NSMutableDictionary<NSString *,NSNumber *> *store;
@property (strong) NSMutableDictionary<NSString *,NSNumber *> *delete;
@end

@implementation CKKSPowerCollection

+ (void)CKKSPowerEvent:(CKKSPowerEvent *)operation zone:(NSString *)zone
{
    SecPLLogRegisteredEvent(@"CKKSSyncing", @{
        @"operation" : operation,
        @"zone" : zone
    });
}

+ (void)CKKSPowerEvent:(CKKSPowerEvent *)operation zone:(NSString *)zone count:(NSUInteger)count
{
    SecPLLogRegisteredEvent(@"CKKSSyncing", @{
        @"operation" : operation,
        @"zone" : zone,
        @"count" : @(count)
    });
}

+ (void)CKKSPowerEvent:(CKKSPowerEvent *)operation count:(NSUInteger)count
{
    SecPLLogRegisteredEvent(@"CKKSSyncing", @{
                                              @"operation" : operation,
                                              @"count" : @(count)
                                              });
}

+ (void)OTPowerEvent:(NSString *)operation
{
    SecPLLogRegisteredEvent(@"OctagonTrust", @{
        @"operation" : operation
    });
}

- (instancetype)init
{
    if ((self = [super init]) != nil) {
        _store = [NSMutableDictionary dictionary];
        _delete = [NSMutableDictionary dictionary];
    }
    return self;
}

- (void)addToStatsDictionary:(NSMutableDictionary *)stats key:(NSString *)key
{
    if(!key) {
        key = @"access-group-missing";
    }
    NSNumber *number = stats[key];
    stats[key] = @([number longValue] + 1);
}

- (void)storedOQE:(CKKSOutgoingQueueEntry *)oqe
{
    [self addToStatsDictionary:_store key:oqe.accessgroup];
}
- (void)deletedOQE:(CKKSOutgoingQueueEntry *)oqe
{
    [self addToStatsDictionary:_delete key:oqe.accessgroup];
}

-(void)summary:(NSString *)operation stats:(NSDictionary<NSString *,NSNumber *> *)stats
{
    for (NSString *accessGroup in stats) {
        SecPLLogRegisteredEvent(@"CKKSSyncing", @{
            @"operation" : operation,
            @"accessgroup" : accessGroup,
            @"items" : stats[accessGroup]
        });
    }
}

- (void)commit
{
    [self summary:@"store" stats:_store];
    [self summary:@"delete" stats:_delete];
}

@end

#endif
