/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
 * CLCachedEntry.cpp - classes representing cached certs and CRLs. 
 *
 * Copyright (c) 2000,2011,2014 Apple Inc. 
 */

#include "CLCachedEntry.h"

/*
 * CLCachedEntry base class constructor. Only job here is to cook up 
 * a handle.
 */
CLCachedEntry::CLCachedEntry()
{
	mHandle = reinterpret_cast<CSSM_HANDLE>(this);
}

CLCachedCert::~CLCachedCert()
{
	delete &mCert;
}

CLCachedCRL::~CLCachedCRL()
{
	delete &mCrl;
}

CLQuery::CLQuery(
	CLQueryType		type,
	const CssmOid	&oid,
	unsigned		numFields,
	bool			isFromCache,
	CSSM_HANDLE		cachedObj) :
		mQueryType(type),
		mFieldId(Allocator::standard()),
		mNextIndex(1),
		mNumFields(numFields),
		mFromCache(isFromCache),
		mCachedObject(cachedObj)
{
	mFieldId.copy(oid);
	mHandle = reinterpret_cast<CSSM_HANDLE>(this);
}	

CLQuery::~CLQuery()
{
	/* mFieldId auto frees */
}
