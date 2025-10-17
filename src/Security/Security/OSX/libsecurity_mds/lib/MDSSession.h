/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#ifndef _MDSSESSION_H_
#define _MDSSESSION_H_  1

#include <security_cdsa_plugin/DatabaseSession.h>
#include <security_cdsa_utilities/handleobject.h>
#include <security_cdsa_utilities/cssmdb.h>
#include <Security/mdspriv.h>
#include "MDSModule.h"
#include "MDSSchema.h"
#include <map>
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/types.h>
#include <list>

namespace Security
{

class MDSSession: public DatabaseSession, public HandleObject
{
    NOCOPY(MDSSession)
public:
    MDSSession (const Guid *inCallerGuid,
                const CSSM_MEMORY_FUNCS &inMemoryFunctions);
    virtual ~MDSSession ();

    void terminate ();
    void install ();
    void uninstall ();

	CSSM_DB_HANDLE dbOpen(const char *dbName, bool batched = false);
		
	// some DatabaseSession routines we need to override
	void DbOpen(const char *DbName,
			const CSSM_NET_ADDRESS *DbLocation,
			CSSM_DB_ACCESS_TYPE AccessRequest,
			const AccessCredentials *AccessCred,
			const void *OpenParameters,
			CSSM_DB_HANDLE &DbHandle);
    CSSM_HANDLE DataGetFirst(CSSM_DB_HANDLE DBHandle,
                             const CssmQuery *Query,
                             CSSM_DB_RECORD_ATTRIBUTE_DATA_PTR Attributes,
                             CssmData *Data,
                             CSSM_DB_UNIQUE_RECORD_PTR &UniqueId);
    void GetDbNames(CSSM_NAME_LIST_PTR &NameList);
    void FreeNameList(CSSM_NAME_LIST &NameList);
    void GetDbNameFromHandle(CSSM_DB_HANDLE DBHandle,
			char **DbName);
	
	// additional public (or private API) methods
	void installFile(const MDS_InstallDefaults *defaults,
		const char *inBundlePath, const char *subdir, const char *file);
	void removeSubservice(const char *guid, uint32 ssid);

    // implement CssmHeap::Allocator
    void *malloc(size_t size)
	{ return mCssmMemoryFunctions.malloc(size); }
    void free(void *addr) _NOEXCEPT
	{ mCssmMemoryFunctions.free(addr); }
	void *realloc(void *addr, size_t size)
	{ return mCssmMemoryFunctions.realloc(addr, size); }

	MDSModule		&module() 	{ return mModule; }
	void removeRecordsForGuid(
		const char *guid,
		CSSM_DB_HANDLE dbHand);

	
	/* 
	 * represents two DB files in any location and state
	 */
	class DbFilesInfo
	{
	public:
		DbFilesInfo(MDSSession &session, const char *dbPath);
		~DbFilesInfo();
		/* these three may not be needed */
		CSSM_DB_HANDLE objDbHand();
		CSSM_DB_HANDLE directDbHand();
		time_t laterTimestamp()			{ return mLaterTimestamp; }

		/* public functions used by MDSSession */
		void updateSystemDbInfo(
			const char *systemPath,			// e.g., /System/Library/Frameworks
			const char *bundlePath);		// e.g., /System/Library/Security
		void removeOutdatedPlugins();
		void updateForBundleDir(
			const char *bundleDirPath);
		void updateForBundle(
			const char *bundlePath);
	private:
		bool lookupForPath(
			const char *path);

		/* object and list to keep track of "to be deleted" records */
		#define MAX_GUID_LEN	64		/* normally 37 */
		class TbdRecord
		{
		public:
			TbdRecord(const CSSM_DATA &guid);
			~TbdRecord() 		{ } 
			const char *guid() 	{ return mGuid; }
		private:
			char mGuid[MAX_GUID_LEN];
		};
		typedef vector<TbdRecord *> TbdVector;

		void checkOutdatedPlugin(
			const CSSM_DATA &pathValue, 
			const CSSM_DATA &guidValue, 
			TbdVector &tbdVector);

		MDSSession &mSession;
		char mDbPath[MAXPATHLEN];
		CSSM_DB_HANDLE mObjDbHand;
		CSSM_DB_HANDLE mDirectDbHand;
		time_t mLaterTimestamp;
	};	/* DbFilesInfo */
private:
    class LockHelper
    {
    private:
        int mFD;

    public:
        LockHelper() : mFD(-1) {}
        ~LockHelper();

        bool obtainLock(
            const char *lockFile,
            int timeout = 0);
	};

	/* given DB file name, fill in fully specified path */
	void dbFullPath(
		const char *dbName,
		char fullPath[MAXPATHLEN+1]);
	
	void updateDataBases();
	
	void clearRecords(CSSM_DB_HANDLE dbHand, const CssmQuery &query);

	bool systemDatabasesPresent(bool purge);
	void createSystemDatabase(
		const char *dbName,
		const RelationInfo *relationInfo,
		unsigned numRelations,
		CSSM_BOOL autoCommit,
		mode_t mode,
		CSSM_DB_HANDLE &dbHand);		// RETURNED
	bool createSystemDatabases(
		CSSM_BOOL autoCommit,
		mode_t mode);
	
	RecursionBlock mUpdating;		// updateDatabases() in progress

    const CssmMemoryFunctions mCssmMemoryFunctions;
    Guid 			mCallerGuid;
    bool 			mCallerGuidPresent;
	
	MDSModule		&mModule;
};

} // end namespace Security

#endif //_MDSSESSION_H_
