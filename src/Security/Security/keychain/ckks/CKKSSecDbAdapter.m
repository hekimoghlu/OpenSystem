/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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

#include <os/transaction_private.h>

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSSecDbAdapter.h"
#include "keychain/securityd/SecItemServer.h"
#include "keychain/securityd/SecItemDb.h"

@interface CKKSSecDbAdapter ()
@property dispatch_queue_t queue;
@end

@implementation CKKSSecDbAdapter

- (instancetype)initWithQueue:(dispatch_queue_t)queue
{
    if((self = [super init])) {
        _queue = queue;
    }
    return self;
}

- (bool)dispatchSyncWithConnection:(SecDbConnectionRef _Nonnull)dbconn
                    readWriteTxion:(BOOL)readWriteTxion
                             block:(CKKSDatabaseTransactionResult (^)(void))block
{
    CFErrorRef cferror = NULL;

    // Take the DB transaction, then get on the local queue.
    // In the case of exclusive DB transactions, we don't really _need_ the local queue, but, it's here for future use.

    SecDbTransactionType txtionType = readWriteTxion ? kSecDbExclusiveRemoteCKKSTransactionType : kSecDbNormalTransactionType;
    bool ret = kc_transaction_type(dbconn, txtionType, &cferror, ^bool{
        __block CKKSDatabaseTransactionResult result = CKKSDatabaseTransactionRollback;

        CKKSSQLInTransaction = true;
        if(readWriteTxion) {
            CKKSSQLInWriteTransaction = true;
        }

        dispatch_sync(self.queue, ^{
            result = block();
        });

        if(readWriteTxion) {
            CKKSSQLInWriteTransaction = false;
        }
        CKKSSQLInTransaction = false;
        return result == CKKSDatabaseTransactionCommit;
    });

    if(cferror) {
        ckkserror_global("ckks", "error doing database transaction, major problems ahead: %@", cferror);
    }
    CFReleaseNull(cferror);
    return ret;
}

- (void)dispatchSyncWithSQLTransaction:(CKKSDatabaseTransactionResult (^)(void))block
{
    // important enough to block this thread. Must get a connection first, though!

    // Please don't jetsam us...
    os_transaction_t transaction = os_transaction_create("com.apple.securityd.ckks");

    CFErrorRef cferror = NULL;
    kc_with_dbt(true, &cferror, ^bool (SecDbConnectionRef dbt) {
        return [self dispatchSyncWithConnection:dbt
                                 readWriteTxion:YES
                                          block:block];

    });
    if(cferror) {
        ckkserror_global("ckks", "error getting database connection, major problems ahead: %@", cferror);
    }
    CFReleaseNull(cferror);

    (void)transaction;
}

- (void)dispatchSyncWithReadOnlySQLTransaction:(void (^)(void))block
{
    // Please don't jetsam us...
    os_transaction_t transaction = os_transaction_create("com.apple.securityd.ckks");

    CFErrorRef cferror = NULL;

    kc_with_dbt(false, &cferror, ^bool (SecDbConnectionRef dbt) {
        return [self dispatchSyncWithConnection:dbt
                                 readWriteTxion:NO
                                          block:^CKKSDatabaseTransactionResult {
            block();
            return CKKSDatabaseTransactionCommit;
        }];

    });
    if(cferror) {
        ckkserror_global("ckks", "error getting database connection, major problems ahead: %@", cferror);
    }
    CFReleaseNull(cferror);

    (void)transaction;
}

- (BOOL)insideSQLTransaction
{
    return CKKSSQLInTransaction;
}


@end

#endif //OCTAGON
