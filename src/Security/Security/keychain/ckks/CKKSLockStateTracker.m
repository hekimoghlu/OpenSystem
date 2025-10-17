/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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

#include <notify.h>
#include <dispatch/dispatch.h>
#include <utilities/SecAKSWrappers.h>

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSResultOperation.h"
#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ckks/CKKSLockStateTracker.h"
#import "keychain/ot/ObjCImprovements.h"

@interface CKKSLockStateTracker ()
@property bool queueIsLocked;
@property dispatch_queue_t queue;
@property NSOperationQueue* operationQueue;
@property NSHashTable<id<CKKSLockStateNotification>> *observers;
@property (assign) int notify_token;

@property (nullable) NSDate* lastUnlockedTime;

@end

@implementation CKKSLockStateTracker

- (instancetype)initWithProvider:(id<CKKSLockStateProviderProtocol>)provider
{
    if((self = [super init])) {
        _lockStateProvider = provider;

        _queue = dispatch_queue_create("lock-state-tracker", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
        _operationQueue = [[NSOperationQueue alloc] init];

        _notify_token = NOTIFY_TOKEN_INVALID;
        _queueIsLocked = true;
        _observers = [NSHashTable weakObjectsHashTable];
        [self resetUnlockDependency];

        WEAKIFY(self);

        // If this is a live server, register with notify
        if(!SecCKKSTestsEnabled()) {
            notify_register_dispatch(kUserKeybagStateChangeNotification, &_notify_token, _queue, ^(int t) {
                STRONGIFY(self);
                [self _onqueueRecheck];
            });
        }

        dispatch_async(_queue, ^{
            STRONGIFY(self);
            [self _onqueueRecheck];
        });
    }
    return self;
}

- (void)dealloc {
    if (_notify_token != NOTIFY_TOKEN_INVALID) {
        notify_cancel(_notify_token);
    }
}

- (bool)isLocked {
    // if checking close after launch ``isLocked'' needs to be blocked on the initial fetch operation
    __block bool locked;
    dispatch_sync(_queue, ^{
        locked = self->_queueIsLocked;
    });
    
    return locked;
}

- (NSDate*)lastUnlockTime {
    // If unlocked, the last unlock time is now. Otherwise, used the cached value.
    __block NSDate* date = nil;
    dispatch_sync(self.queue, ^{
        if(self.queueIsLocked) {
            date = self.lastUnlockedTime;
        } else {
            date = [NSDate date];
            self.lastUnlockedTime = date;
        }
    });
    return date;
}

-(NSString*)description {
    bool isLocked = self.isLocked;
    return [NSString stringWithFormat: @"<CKKSLockStateTracker: %@ last:%@>",
            isLocked ? @"locked" : @"unlocked",
            isLocked ? self.lastUnlockedTime : @"now"];
}

-(void)resetUnlockDependency {
    if(self.unlockDependency == nil || ![self.unlockDependency isPending]) {
        CKKSResultOperation* op = [CKKSResultOperation named:@"keybag-unlocked-dependency" withBlock: ^{
            ckksinfo_global("ckks", "Keybag unlocked");
        }];
        op.descriptionErrorCode = CKKSResultDescriptionPendingUnlock;
        self.unlockDependency = op;
    }
}

-(void)_onqueueRecheck {
    dispatch_assert_queue(self.queue);

    static bool first = true;
    bool wasLocked = self.queueIsLocked;
    self.queueIsLocked = [self.lockStateProvider queryAKSLocked];

    if(wasLocked != self.queueIsLocked || first) {
        first = false;
        if(self.queueIsLocked) {
            // We're locked now.
            [self resetUnlockDependency];

            if(wasLocked) {
                self.lastUnlockedTime = [NSDate date];
            }
        } else {
            [self.operationQueue addOperation: self.unlockDependency];
            self.unlockDependency = nil;
            self.lastUnlockedTime = [NSDate date];
        }

        bool isUnlocked = (self.queueIsLocked == false);
        for (id<CKKSLockStateNotification> observer in _observers) {
            __strong typeof(observer) strongObserver = observer;
            if (strongObserver == nil) {
                return;
            }
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
                [strongObserver lockStateChangeNotification:isUnlocked];
            });
        }
    }
}

-(void)recheck {
    dispatch_sync(self.queue, ^{
        [self _onqueueRecheck];
    });
}

- (bool)lockedError:(NSError *)error
{
    bool isLockedError = error.code == errSecInteractionNotAllowed &&
        ([error.domain isEqualToString:@"securityd"] || [error.domain isEqualToString:(__bridge NSString*)kSecErrorDomain]);
    return isLockedError;
}

- (bool)checkErrorChainForLockState:(NSError*)error
{
    while(error != nil) {
        if([self lockedError:error]) {
            return true;
        }

        error = error.userInfo[NSUnderlyingErrorKey];
    }

    return false;
}

-(bool)isLockedError:(NSError *)error {
    bool isLockedError = [self checkErrorChainForLockState:error];

    /*
     * If we are locked, and the the current lock state track disagree, lets double check
     * if we are actually locked since we might have missed a lock state notification,
     * and cause spinning to happen.
     *
     * We don't update the local variable, since the error code was a locked error.
     */
    if (isLockedError) {
        dispatch_sync(self.queue, ^{
            if (self.queueIsLocked == false) {
                [self _onqueueRecheck];
            }
        });
    }
    return isLockedError;
}

-(void)addLockStateObserver:(id<CKKSLockStateNotification>)object
{
    dispatch_async(self.queue, ^{
        [self->_observers addObject:object];
        bool isUnlocked = (self.queueIsLocked == false);
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
            [object lockStateChangeNotification:isUnlocked];
        });
    });
}

+ (CKKSLockStateTracker*)globalTracker
{
    static CKKSLockStateTracker* tracker;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        tracker = [[CKKSLockStateTracker alloc] initWithProvider:[[CKKSActualLockStateProvider alloc] init]];
    });
    return tracker;
}


@end

@implementation CKKSActualLockStateProvider
- (instancetype)init {
    if((self = [super init])) {
    }
    return self;
}

- (BOOL)queryAKSLocked {
    CFErrorRef aksError = NULL;
    bool locked = true;

    if(!SecAKSGetIsLocked(g_keychain_keybag, &locked, &aksError)) {
        ckkserror_global("ckks", "error querying lock state: %@", aksError);
        CFReleaseNull(aksError);
    }

    return locked ? YES : NO;
}
@end

#endif // OCTAGON
