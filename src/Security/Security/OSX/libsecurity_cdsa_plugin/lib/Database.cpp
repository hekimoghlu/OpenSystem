/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#ifdef __MWERKS__
#define _CPP_DATABASE
#endif
#include <security_cdsa_plugin/Database.h>
#include <Security/cssmerr.h>
#include <security_cdsa_plugin/DbContext.h>
#include <memory>

DatabaseManager::DatabaseManager ()
{
}

DatabaseManager::~DatabaseManager ()
{
}

Database *
DatabaseManager::get (const DbName &inDbName)
{
    StLock<Mutex> _(mDatabaseMapLock);
    DatabaseMap::iterator anIterator = mDatabaseMap.find (inDbName);
    if (anIterator == mDatabaseMap.end())
    {
        unique_ptr<Database> aDatabase(make(inDbName));
        mDatabaseMap.insert(DatabaseMap::value_type(aDatabase->mDbName, aDatabase.get()));
        return aDatabase.release();
    }

    return anIterator->second;
}

void
DatabaseManager::removeIfUnused(Database &inDatabase)
{
    StLock<Mutex> _(mDatabaseMapLock);
    if (!inDatabase.hasDbContexts()) {
        mDatabaseMap.erase(inDatabase.mDbName);
		delete &inDatabase;
	}
}

DbContext &
DatabaseManager::dbOpen(DatabaseSession &inDatabaseSession,
                        const DbName &inDbName,
                        CSSM_DB_ACCESS_TYPE inAccessRequest,
                        const AccessCredentials *inAccessCred,
                        const void *inOpenParameters)
{
    Database &aDatabase = *get(inDbName);
    try
    {
        return aDatabase._dbOpen(inDatabaseSession, inAccessRequest, inAccessCred, inOpenParameters);
    }
    catch (...)
    {
        removeIfUnused(aDatabase);
        throw;
    }
}

DbContext &
DatabaseManager::dbCreate(DatabaseSession &inDatabaseSession,
                          const DbName &inDbName,
                          const CSSM_DBINFO &inDBInfo,
                          CSSM_DB_ACCESS_TYPE inAccessRequest,
                          const CSSM_RESOURCE_CONTROL_CONTEXT *inCredAndAclEntry,
                          const void *inOpenParameters)
{
    Database &aDatabase = *get(inDbName);
    try
    {
        return aDatabase._dbCreate(inDatabaseSession, inDBInfo, inAccessRequest,
                                   inCredAndAclEntry, inOpenParameters);
    }
    catch (...)
    {
        removeIfUnused(aDatabase);
        throw;
    }
}

// Delete a DbContext instance created by calling dbOpen or dbCreate.
void
DatabaseManager::dbClose(DbContext &inDbContext)
{
    Database &aDatabase = inDbContext.mDatabase;
    aDatabase._dbClose(inDbContext);
    removeIfUnused(aDatabase);
}

// Delete a database.
void
DatabaseManager::dbDelete(DatabaseSession &inDatabaseSession,
                          const DbName &inDbName,
                          const AccessCredentials *inAccessCred)
{
    Database &aDatabase = *get(inDbName);
    try
    {
        aDatabase.dbDelete(inDatabaseSession, inAccessCred);
    }
    catch (...)
    {
        removeIfUnused(aDatabase);
        throw;
    }

    removeIfUnused(aDatabase);
}

// List all available databases.
CSSM_NAME_LIST_PTR
DatabaseManager::getDbNames(DatabaseSession &inDatabaseSession)
{
    CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED);
}

void
DatabaseManager::freeNameList(DatabaseSession &inDatabaseSession,
                  CSSM_NAME_LIST &inNameList)
{
    CssmError::throwMe(CSSM_ERRCODE_FUNCTION_NOT_IMPLEMENTED);
}

// Start of Database implementation.

Database::Database (const DbName &inDbName)
: mDbName(inDbName)
{
}

Database::~Database ()
{
}

bool
Database::hasDbContexts()
{
    StLock<Mutex> _(mDbContextSetLock);
    return !mDbContextSet.empty();
}

DbContext &
Database::_dbOpen(DatabaseSession &inDatabaseSession,
                  CSSM_DB_ACCESS_TYPE inAccessRequest,
                  const AccessCredentials *inAccessCred,
                  const void *inOpenParameters)
{
    unique_ptr<DbContext>aDbContext(makeDbContext(inDatabaseSession,
                                                inAccessRequest,
                                                inAccessCred,
                                                inOpenParameters));
    {
        StLock<Mutex> _(mDbContextSetLock);
        mDbContextSet.insert(aDbContext.get());
        // Release the mDbContextSetLock
    }

    try
    {
        dbOpen(*aDbContext);
    }
    catch (...)
    {
        StLock<Mutex> _(mDbContextSetLock);
        mDbContextSet.erase(aDbContext.get());
        throw;
    }

    return *aDbContext.release();
}

DbContext &
Database::_dbCreate(DatabaseSession &inDatabaseSession,
                    const CSSM_DBINFO &inDBInfo,
                    CSSM_DB_ACCESS_TYPE inAccessRequest,
                    const CSSM_RESOURCE_CONTROL_CONTEXT *inCredAndAclEntry,
                    const void *inOpenParameters)
{
    unique_ptr<DbContext>aDbContext(makeDbContext(inDatabaseSession,
                                                inAccessRequest,
                                                (inCredAndAclEntry
												 ? AccessCredentials::optional(inCredAndAclEntry->AccessCred)
												 : NULL),
                                                inOpenParameters));
    {
        StLock<Mutex> _(mDbContextSetLock);
        mDbContextSet.insert(aDbContext.get());
        // Release the mDbContextSetLock
    }

    try
    {
        dbCreate(*aDbContext, inDBInfo,
                 inCredAndAclEntry ? &inCredAndAclEntry->InitialAclEntry : NULL);
    }
    catch (...)
    {
        StLock<Mutex> _(mDbContextSetLock);
        mDbContextSet.erase(aDbContext.get());
        throw;
    }

    return *aDbContext.release();
}

void
Database::_dbClose(DbContext &dbContext)
{
    StLock<Mutex> _(mDbContextSetLock);
    mDbContextSet.erase(&dbContext);
    if (mDbContextSet.empty())
        dbClose();
}
