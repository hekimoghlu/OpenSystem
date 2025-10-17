/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
//
// mdsclient - friendly interface to CDSA MDS API
//
// It is useful to think of the mdsclient interface as "slightly below" the
// rest of the cdsa_client layer. It does not actually call into CSSM (we
// consider MDS as a separate facility, "slightly lower" than CSSM as well).
// This means that you can use mdsclient without creating a binary dependency
// on CSSM, and thus Security.framework.
//

#ifndef _H_CDSA_CLIENT_MDSCLIENT
#define _H_CDSA_CLIENT_MDSCLIENT

#include <security_utilities/threading.h>
#include <security_utilities/globalizer.h>
#include <security_utilities/refcount.h>
#include <security_cdsa_utilities/cssmalloc.h>
#include <security_cdsa_utilities/cssmpods.h>
#include <security_cdsa_utilities/cssmerrors.h>
#include <security_cdsa_utilities/cssmdb.h>
#include <security_cdsa_client/dliterators.h>
#include <Security/mdspriv.h>
#include <Security/mds_schema.h>


namespace Security {
namespace MDSClient {

// import query sublanguage classes into MDSClient namespace
using CssmClient::Attribute;
using CssmClient::Query;
using CssmClient::Record;
using CssmClient::Table;


//
// A singleton for the MDS itself.
// This is automatically created as a ModuleNexus when needed.
// You can reset() it to release resources.
// Don't make your own.
//
class Directory : public MDS_FUNCS, public CssmClient::DLAccess {
public:
	Directory();
	virtual ~Directory();

	MDS_HANDLE mds() const { return mCDSA.DLHandle; }
	const MDS_DB_HANDLE &cdsa() const;

public:
	CSSM_HANDLE dlGetFirst(const CSSM_QUERY &query,
		CSSM_DB_RECORD_ATTRIBUTE_DATA &attributes, CSSM_DATA *data,
		CSSM_DB_UNIQUE_RECORD *&id);
	bool dlGetNext(CSSM_HANDLE handle,
		CSSM_DB_RECORD_ATTRIBUTE_DATA &attributes, CSSM_DATA *data,
		CSSM_DB_UNIQUE_RECORD *&id);
	void dlAbortQuery(CSSM_HANDLE handle);
	void dlFreeUniqueId(CSSM_DB_UNIQUE_RECORD *id);
	void dlDeleteRecord(CSSM_DB_UNIQUE_RECORD *id);
	Allocator &allocator();
	
public:
	// not for ordinary use - system administration only
	void install();						// system default install/regenerate
	void install(const MDS_InstallDefaults *defaults, // defaults
		const char *path,				// path to bundle (NULL -> main)
		const char *subdir = NULL,		// subdirectory in Resources (NULL -> all)
		const char *file = NULL);		// individual file (NULL -> all)
	void uninstall(const char *guid, uint32 ssid);

private:
	mutable MDS_DB_HANDLE mCDSA;		// CDSA database handle
	mutable Mutex mInitLock;			// interlock for lazy DB open
	CssmAllocatorMemoryFunctions mMemoryFunctions;
	Guid mCallerGuid;					//@@@ fake/unused
};

extern ModuleNexus<Directory> mds;


} // end namespace MDSClient
} // end namespace Security

#endif // _H_CDSA_CLIENT_MDSCLIENT
