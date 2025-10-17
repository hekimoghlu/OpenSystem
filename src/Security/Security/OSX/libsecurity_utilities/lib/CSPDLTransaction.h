/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#ifndef _H_CSPDLTRANSACTION
#define _H_CSPDLTRANSACTION

#include <TargetConditionals.h>

#if TARGET_OS_OSX

#include <security_cdsa_client/dlclient.h>

//
// This class performs a transaction on a CSPDL database.
//
// If commit() has not yet been called when the object goes out of scope, the transaction will roll back instead (exceptions will be swallowed).
//
// Nesting transactions will likely work, but isn't recommended.
//
class DLTransaction {
public:
    DLTransaction(CSSM_DL_DB_HANDLE dldbh);

    ~DLTransaction();

    // Everything has gone right; this transaction will commit.
    // If you don't call this, the transaction will roll back when the object goes out of scope.
    // Might throw on error.
    void commit();

protected:
    DLTransaction();

    // Note: disables autocommit using the dldbh
    void initialize();

    // Call rollback if necessary. Never throws.
    void finalize();

    // Rolls back database transactions. Might throw.
    void rollback();

    CSSM_DL_DB_HANDLE mDldbh;

    bool mSuccess;
    bool mFinalized;

    CSSM_BOOL mAutoCommit;
};

class CSPDLTransaction : public DLTransaction {
public:
    CSPDLTransaction(Security::CssmClient::Db& db);
    ~CSPDLTransaction();

private:
    Security::CssmClient::Db& mDb;
};

#endif //TARGET_OS_OSX

#endif // _H_CSPDLTRANSACTION
