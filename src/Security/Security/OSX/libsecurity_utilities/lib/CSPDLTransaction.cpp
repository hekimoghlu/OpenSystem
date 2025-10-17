/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#include "CSPDLTransaction.h"
#include <Security/SecBasePriv.h>
#include <syslog.h>

#if TARGET_OS_OSX

DLTransaction::DLTransaction(CSSM_DL_DB_HANDLE dldbh)
    : mDldbh(dldbh), mSuccess(false), mFinalized(false), mAutoCommit(CSSM_TRUE) {
    initialize();
}

DLTransaction::DLTransaction()
    : mSuccess(false), mFinalized(false), mAutoCommit(CSSM_TRUE) {
}

void DLTransaction::initialize() {
    // Turn off autocommit on the underlying DL and remember the old state.
    Security::CssmClient::ObjectImpl::check(CSSM_DL_PassThrough(mDldbh,
                CSSM_APPLEFILEDL_TOGGLE_AUTOCOMMIT,
                NULL, reinterpret_cast<void **>(&mAutoCommit)));
}

DLTransaction::~DLTransaction() {
    finalize();
}

void DLTransaction::commit() {
    // Commit the transaction, and throw if it fails

    // If autocommit wasn't on on the database when we started, don't
    // actually commit. There might be something else going on...
    if(mAutoCommit) {
        Security::CssmClient::ObjectImpl::check(CSSM_DL_PassThrough(mDldbh, CSSM_APPLEFILEDL_COMMIT, NULL, NULL));
        CSSM_DL_PassThrough(mDldbh, CSSM_APPLEFILEDL_TOGGLE_AUTOCOMMIT, reinterpret_cast<const void *>(mAutoCommit), NULL);
    }

    // Throwing above means this wasn't a success and we're not finalized. On exit, we'll roll back the transaction.
    mSuccess = true;
    mFinalized = true;
}

void DLTransaction::rollback() {
    // If autocommit wasn't on on the database when we started, don't
    // actually roll back. There might be something else going on...
    if(mAutoCommit) {
        CSSM_DL_PassThrough(mDldbh, CSSM_APPLEFILEDL_ROLLBACK, NULL, NULL);
        CSSM_DL_PassThrough(mDldbh, CSSM_APPLEFILEDL_TOGGLE_AUTOCOMMIT,
                            reinterpret_cast<const void *>(mAutoCommit), NULL);
    }
}

void DLTransaction::finalize() {
    if(mFinalized) {
        return;
    }

    // if this transaction was not a success, roll back.
    if(!mSuccess) {
        // Note that we're likely (but not necessarily) unwinding the stack for an exception right now.
        // (If this transaction succeeded, we wouldn't be here. So, it failed, and this code likes to fail with exceptions.)
        // If this throws an exception, we might crash the whole process.
        // Swallow exceptions whole, but log them aggressively.
        try {
            rollback();
        } catch(CssmError cssme) {
            const char* errStr = cssmErrorString(cssme.error);
            secnotice("integrity", "caught CssmError during transaction rollback: %d %s", (int) cssme.error, errStr);
            syslog(LOG_ERR, "ERROR: failed to rollback keychain transaction: %d %s", (int) cssme.error, errStr);
        }
    }
    mFinalized = true;
}


CSPDLTransaction::CSPDLTransaction(Security::CssmClient::Db& db)
    : DLTransaction(), mDb(db) {
    // Get the handle of the DL underlying this CSPDL.
    mDb->passThrough(CSSM_APPLECSPDL_DB_GET_HANDLE, NULL,
            reinterpret_cast<void **>(&mDldbh));

    initialize();
}

CSPDLTransaction::~CSPDLTransaction() {
}

#endif //TARGET_OS_OSX
