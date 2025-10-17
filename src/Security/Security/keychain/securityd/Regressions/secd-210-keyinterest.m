/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
//  secd-210-keyinterest.m
//  Security
//
//  Created by Mitch Adler on 10/31/16.
//
//

#import <Foundation/Foundation.h>

#include "secd_regressions.h"

#import "CKDStore.h"
#import "CKDKVSProxy.h"
#import "CKDSimulatedStore.h"
#import "CKDSimulatedAccount.h"
#import "CKDAKSLockMonitor.h"
#include "SOSCloudKeychainConstants.h"
#include "SOSAccountTesting.h"


#if SOS_ENABLED

@interface CKDSimulatedLockMonitor : NSObject<CKDLockMonitor>

@property (readwrite) BOOL unlockedSinceBoot;
@property (readwrite) BOOL locked;

@property (weak) NSObject<CKDLockListener>* listener;

+ (instancetype) monitor;

- (instancetype) init;

- (void) recheck;

- (void) notifyListener;
- (void) connectTo: (NSObject<CKDLockListener>*) listener;

- (void) lock;
- (void) unlock;

@end


@implementation CKDSimulatedLockMonitor

+ (instancetype) monitor {
    return [[CKDSimulatedLockMonitor alloc] init];
}

- (instancetype) init {
    if ((self = [super init])) {
        _locked = true;
        _unlockedSinceBoot = false;
    }

    [self notifyListener];

    return self;
}

- (void) recheck {
}

- (void) notifyListener {
    // Take a strong reference:
    __strong __typeof(self.listener) listener = self.listener;

    if (listener) {
        if (self.locked) {
            [listener locked];
        } else {
            [listener unlocked];
        }
    }
}

- (void) connectTo: (NSObject<CKDLockListener>*) listener {
    self.listener = listener;
    [self notifyListener];
}

- (void) lock {
    self.locked = true;
    [self notifyListener];
}
- (void) unlock {
    self.locked = false;
    self.unlockedSinceBoot = true;
    [self notifyListener];
}


@end

@interface UbiqitousKVSProxy (Testing)
- (void) flush;
@end

@implementation UbiqitousKVSProxy (Testing)
- (void) flush {
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    
    [self doAfterFlush:^{
        dispatch_semaphore_signal(sema);
    }];

    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
}
@end

static void tests(void) {
    CKDSimulatedStore* store = [CKDSimulatedStore simulatedInterface];
    CKDSimulatedAccount* account = [[CKDSimulatedAccount alloc] init];
    CKDSimulatedLockMonitor* monitor = [CKDSimulatedLockMonitor monitor];

    NSString * testKey = @"TestKey";

    UbiqitousKVSProxy * proxy = [UbiqitousKVSProxy withAccount:account
                                                         store:store
                                                   lockMonitor:monitor
                                                   persistence:[NSURL fileURLWithPath:@"/tmp/kvsPersistenceTestFile"]];

    NSDictionary* interests = @{ [NSString stringWithUTF8String:kMessageKeyParameter]:@{ @"UnlockedKeys":@[ testKey ] } };
    NSString* accountID = @"Account1";

    dispatch_sync([proxy ckdkvsproxy_queue], ^{
        [proxy registerKeys:interests forAccount:accountID];
    });

    is([[account extractKeyChanges] count], (NSUInteger)0, "No changes yet");

    [store remoteSetObject:@1 forKey:testKey];
    [proxy flush];

    is([[account extractKeyChanges] count], (NSUInteger)0, "Still none while locked");

    [monitor unlock];
    [proxy flush];

    is([[account extractKeyChanges] count], (NSUInteger)1, "Notified after unlock");

    [monitor lock];
    [monitor unlock];
    [proxy flush];

    is([[account extractKeyChanges] count], (NSUInteger)0, "lock unlock and nothing changes");

    [store remoteSetObject:@2 forKey:testKey];
    [proxy flush];

    {
        NSDictionary<NSString*, NSObject*> *changes = [account extractKeyChanges];
        is([changes count], (NSUInteger)1, "lock, nothing changes");
        is(changes[testKey], @2, "Sent second value");
    }

    [monitor lock];
    [store remoteSetObject:@3 forKey:testKey];
    [proxy flush];

    is([[account extractKeyChanges] count], (NSUInteger)0, "Changes to Unlocked not when locked");

    [monitor unlock];
    [proxy flush];

    {
        NSDictionary<NSString*, NSObject*> *changes = [account extractKeyChanges];
        is([changes count], (NSUInteger)1, "Change defered to after unlock");
        is(changes[testKey], @3, "Correct value");
    }

    dispatch_sync([proxy ckdkvsproxy_queue], ^{
        [proxy registerKeys:interests forAccount:accountID];
    });
    [proxy flush];

    is([[account extractKeyChanges] count], (NSUInteger)0, "Same interests, no new data");

    dispatch_sync([proxy ckdkvsproxy_queue], ^{
        [proxy registerKeys:interests forAccount:@"different"];
    });
    [proxy flush];

    {
        NSDictionary<NSString*, NSObject*> *changes = [account extractKeyChanges];
        is([changes count], (NSUInteger)1, "New account, same interests, new data");
        is(changes[testKey], @3, "Latest value for new data");
    }

}

#endif

int secd_210_keyinterest(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(12);
    enableSOSCompatibilityForTests();
    OctagonSetSOSFeatureEnabled(true);
    tests();
#else
    plan_tests(0);
#endif
    return 0;
}
