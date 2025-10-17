/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#ifndef _DBCONTEXT_H_
#define _DBCONTEXT_H_  1

#include <security_cdsa_plugin/Database.h>
#include <security_cdsa_utilities/handleobject.h>

#ifdef _CPP_DBCONTEXT
# pragma export on
#endif

namespace Security
{

class DatabaseSession;

class DbContext : public HandleObject
{
	NOCOPY(DbContext)
public:
    Database &mDatabase;
    DatabaseSession &mDatabaseSession;

    DbContext(Database &inDatabase,
              DatabaseSession &inDatabaseSession,
              CSSM_DB_ACCESS_TYPE inAccessRequest,
              const CSSM_ACCESS_CREDENTIALS *inAccessCred);

    virtual ~DbContext();

    CSSM_HANDLE
    dataGetFirst(const CssmQuery *inQuery,
                      CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR inoutAttributes,
                      CssmData *inoutData,
                      CSSM_DB_UNIQUE_RECORD_PTR &outUniqueRecord);

    void
    dataGetNext(CSSM_HANDLE inResultsHandle,
                CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR inoutAttributes,
                CssmData *inoutData,
                CSSM_DB_UNIQUE_RECORD_PTR &outUniqueRecord);

    void
    dataAbortQuery(CSSM_HANDLE inResultsHandle);
private:
    CSSM_DB_ACCESS_TYPE mAccessRequest;
    CSSM_ACCESS_CREDENTIALS *mAccessCred;
    //typedef set<DbQuery *> DbQuerySet;
    //DbQuerySet mDbQuerySet;
    //Mutex mDbQuerySetLock;
};

} // end namespace Security

#ifdef _CPP_DBCONTEXT
# pragma export off
#endif

#endif //_DBCONTEXT_H_
