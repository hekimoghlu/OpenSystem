/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
// PolicyCursor.h
//
#ifndef _SECURITY_POLICYCURSOR_H_
#define _SECURITY_POLICYCURSOR_H_

#include <Security/SecPolicySearch.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <Security/mds.h>
#include <Security/mds_schema.h>
#include <security_utilities/seccfobject.h>
#include "SecCFTypes.h"

namespace Security
{

namespace KeychainCore
{

class Policy;

class PolicyCursor : public SecCFObject
{
    NOCOPY(PolicyCursor)
public:
	SECCFFUNCTIONS(PolicyCursor, SecPolicySearchRef, errSecInvalidSearchRef, gTypes().PolicyCursor)

	PolicyCursor(const CSSM_OID* oid, const CSSM_DATA* value);
	virtual ~PolicyCursor() _NOEXCEPT;
	bool next(SecPointer<Policy> &policy);
	bool oidProvided() { return mOidGiven; }

	static void policy(const CSSM_OID* oid, SecPointer<Policy> &policy);

private:
    //CFArrayRef	 mKeychainSearchList;
    //SecKeyUsage  mKeyUsage;
    //SecPolicyRef mPolicy;
    CssmAutoData		mOid;
    bool				mOidGiven;
    // value ignored (for now?)

#if 1	// quick version -- using built-in policy list

    int					mSearchPos;	// next untried table entry

#else	// MDS version -- later
    bool				mFirstLookup;

    //
    // Initialization
    //
	MDS_HANDLE			mMdsHand;
	CSSM_DB_HANDLE		mDbHand;
	//
    // Used for searching (lookups)
    //
	MDS_DB_HANDLE		mObjDlDb;
	MDS_DB_HANDLE		mCdsaDlDb;
	MDS_FUNCS*			mMdsFuncs;
#endif

	Mutex				mMutex;
};

} // end namespace KeychainCore

} // end namespace Security

#endif // !_SECURITY_POLICYCURSOR_H_
