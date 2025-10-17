/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#import "SOSEnsureBackup.h"
#include <utilities/debugging.h>

#if OCTAGON
#import "keychain/ckks/CKKSLockStateTracker.h"
#import "keychain/ckks/NSOperationCategories.h"
#include "keychain/SecureObjectSync/SOSAccount.h"

static NSOperationQueue *backupOperationQueue;
static CKKSLockStateTracker *lockStateTracker;

void SOSEnsureBackupWhileUnlocked(void) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        backupOperationQueue = [NSOperationQueue new];
        lockStateTracker = [CKKSLockStateTracker globalTracker];
    });

    // CKKSLockStateTracker does not use @synchronized(self). If it ever starts to this needs to be updated.
    @synchronized(lockStateTracker) {
        if ([backupOperationQueue operationCount] > 0) {
            secnotice("engine", "SOSEnsureBackup: Backup already scheduled for next unlock");
        } else {
            secnotice("engine", "SOSEnsureBackup: Scheduling a backup for next unlock");
            NSBlockOperation *backupOperation = [NSBlockOperation blockOperationWithBlock:^{
                secnotice("engine", "Performing keychain backup after unlock because backing up while locked failed");
                SOSAccount *account = (__bridge SOSAccount *)(SOSKeychainAccountGetSharedAccount());

                if(!account) {
                    secnotice("ckks", "Failed to get account object");
                    return;
                }

                [account performTransaction:^(SOSAccountTransaction *transaction) {
                    CFErrorRef error = NULL;
                    NSSet* set = CFBridgingRelease(SOSAccountCopyBackupPeersAndForceSync(transaction, &error));
                    if (set) {
                        secnotice("engine", "SOSEnsureBackup: SOS made a backup of views: %@", set);
                    } else {
                        secerror("engine: SOSEnsureBackup: encountered an error while making backup (%@)", error);
                    }

                    CFReleaseNull(error);
                }];
            }];
            [backupOperation addNullableDependency:lockStateTracker.unlockDependency];
            [backupOperationQueue addOperation:backupOperation];
        }
    }
}
#else
void SOSEnsureBackupWhileUnlocked(void) {
    secnotice("engine", "SOSEnsureBackup not available on this platform");
}
#endif
