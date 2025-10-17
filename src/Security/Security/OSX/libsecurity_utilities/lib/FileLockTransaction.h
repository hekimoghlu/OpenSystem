/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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
#ifndef FileLockTransaction_h
#define FileLockTransaction_h

#if TARGET_OS_OSX

#include <security_cdsa_client/dlclient.h>
//
// This class performs a file lock transaction on a Cssm Db object.
//
// It will attempt to take the file lock upon creation.
//
// It will release the file lock upon destruction OR calling finalize().
//
// If you have called success(), it will tell the file lock transaction to commit
// otherwise, it will tell the file lock transaction to roll back.
//
// If you call setDeleteOnFailure(), and the transaction would normally roll
// back, this transaction will instead delete the Db's underlying file.
//
class FileLockTransaction {
public:
    FileLockTransaction(Security::CssmClient::Db& db);

    ~FileLockTransaction();

    // Everything has gone right; this transaction will commit.
    // If you don't call this, the transaction will roll back.
    void success();

    // Commit or rollback as appropriate
    void finalize();

    // After calling this method, if this class attempts to roll back the
    // transaction, it will also attempt to delete the database file.
    void setDeleteOnFailure();

protected:
    // Actually toggle autocommit using the dldbh
    void initialize();

    Security::CssmClient::Db mDb;

    bool mSuccess;
    bool mFinalized;
    bool mDeleteOnFailure;
};

#endif // TARGET_OS_OSX

#endif
