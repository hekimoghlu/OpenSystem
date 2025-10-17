/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include <TargetConditionals.h>
#if TARGET_OS_OSX

#include "FileLockTransaction.h"
#include <Security/SecBasePriv.h>
#include <syslog.h>

FileLockTransaction::FileLockTransaction(Security::CssmClient::Db& db)
    : mDb(db), mSuccess(false), mFinalized(false), mDeleteOnFailure(false) {
    initialize();
}

void FileLockTransaction::initialize() {
    mDb->takeFileLock();
}

FileLockTransaction::~FileLockTransaction() {
    finalize();
}

void FileLockTransaction::success() {
    mSuccess = true;
}

void FileLockTransaction::setDeleteOnFailure() {
    mDeleteOnFailure = true;
}

void FileLockTransaction::finalize() {
    if(mFinalized) {
        return;
    }

    // if this transaction was a success, commit. Otherwise, roll back.
    if(mSuccess) {
        mDb->releaseFileLock(true);
    } else {
        // This is a failure.

        // Note that we're likely (but not necessarily) unwinding the stack for an exception right now.
        // (If this transaction succeeded, we wouldn't be here. So, it failed, and this code likes to fail with exceptions.)
        // If this throws an exception, we might crash the whole process.
        // Swallow exceptions whole, but log them aggressively.
        try {
            if(mDeleteOnFailure) {
                mDb->deleteFile();
            }
            mDb->releaseFileLock(false);
        } catch(CssmError cssme) {
            const char* errStr = cssmErrorString(cssme.error);
            secnotice("integrity", "caught CssmError during transaction rollback: %d %s", (int) cssme.error, errStr);
            syslog(LOG_ERR, "ERROR: failed to rollback keychain transaction: %d %s", (int) cssme.error, errStr);
        }
    }
    mFinalized = true;
}

#endif //TARGET_OS_OSX
