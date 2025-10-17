/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
/*
 * CLCachedEntry.h - classes representing cached certs and CRLs. 
 *
 * Copyright (c) 2000,2011,2014 Apple Inc. 
 */

#ifndef	_APPLE_X509_CL_CACHED_ENTRY_H_
#define _APPLE_X509_CL_CACHED_ENTRY_H_

#include <Security/cssmtype.h>
#include <security_utilities/utilities.h>
#include <security_cdsa_utilities/cssmdata.h>
#include "DecodedCert.h"
#include "DecodedCrl.h"

/* 
 * There is one of these per active cached object (cert or CRL). 
 * AppleX509CLSession keeps a map of these in cacheMap. 
 */
class CLCachedEntry 
{
public:
	CLCachedEntry();
	virtual ~CLCachedEntry() { }
	CSSM_HANDLE		handle() { return mHandle; }
private:
	CSSM_HANDLE		mHandle;	
};

class CLCachedCert : public CLCachedEntry
{
public:
	CLCachedCert(
		DecodedCert &c) : mCert(c) { }
	~CLCachedCert();
	DecodedCert	&cert()	{ return mCert; }
private:
	/* decoded NSS format */
	DecodedCert &mCert;
};

class CLCachedCRL : public CLCachedEntry
{
public:
	CLCachedCRL(
		DecodedCrl &c) : mCrl(c) { }
	~CLCachedCRL();
	DecodedCrl	&crl()	{ return mCrl; }
private:
	/* decoded NSS format */
	DecodedCrl &mCrl;
};

/*
 * An active query, always associated with a CLCachedEntry.
 * AppleX509CLSession keeps a map of these in queryMap. 
 *
 * In the case of a CLCachedEntry created by an explicit {Cert,CRL}Cache op,
 * there can be multiple queries active for a given cached cert. In
 * the *GetFirst*FieldValue case, there is a one-to-one relationship between
 * the CLQUery and its associated cached object.
 *
 * Out of paranoia in the {Cert,CRL}Cache case, we store the handle of 
 * the associated cached object, not a ref to the object, in case the
 * cached object has been deleted via *AbortCache. We could ref count,
 * but that would require a lock in CLCachedEntry...looking up an object
 * in the session's cache map should not be too expensive. 
 */
 
typedef enum {
	CLQ_Cert = 1,
	CLQ_CRL
} CLQueryType;

class CLQuery 
{
public:
	CLQuery(
		CLQueryType		type,
		const CssmOid	&oid,
		unsigned		numFields,
		bool			isFromCache,
		CSSM_HANDLE		cachedObj);			
		
	~CLQuery();
	
	/*  
	 * Accessors - all member variables are invariant after creation, except 
	 * for nextIndex which can only increment
	 */
	CLQueryType		queryType() 	{ return mQueryType; }
	const CssmOid	&fieldId()		{ return mFieldId; }
	unsigned 		nextIndex()		{ return mNextIndex; }
	void			incrementIndex(){ mNextIndex++; }
	unsigned 		numFields() 	{ return mNumFields; }
	bool			fromCache()		{ return mFromCache; }
	CSSM_HANDLE		cachedObject()	{ return mCachedObject; }
	CSSM_HANDLE		handle()		{ return mHandle;}

private:
	CLQueryType		mQueryType;
	CssmAutoData 	mFieldId;		// thing we're searching for - may be empty
	unsigned 		mNextIndex;		// index of next find op 
	unsigned 		mNumFields;		// total available 
	bool			mFromCache;		// true : via CertGetFirstCachedFieldValue
									// false : via CertGetFirstFieldValue
	CSSM_HANDLE		mCachedObject;	// of our associated cached cert/CRL
	CSSM_HANDLE		mHandle;		// ours
};

#endif	/* _APPLE_X509_CL_CACHED_ENTRY_H_ */
