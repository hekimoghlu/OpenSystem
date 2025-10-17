/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
#include <TargetConditionals.h>
#if TARGET_OS_OSX

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#include "SecNssCoder.h"
#include <Security/cssmerr.h>
#include <security_utilities/utilities.h>
#include <security_asn1/secasn1.h>
#include <string.h>
#include <security_utilities/simulatecrash_assert.h>
#pragma clang diagnostic pop

#ifdef	NDEBUG
#define THROW_ENABLE	1
#else
/* disable link against Security framework when true */
#define THROW_ENABLE	0
#endif

#if		THROW_ENABLE
#define THROW_ERROR	Security::CssmError::throwMe(CSSMERR_CSSM_MEMORY_ERROR)
#else
#define THROW_ERROR
#endif

SecNssCoder::SecNssCoder(PRUint32 chunkSize /* = SNC_CHUNKSIZE_DEF */)
	: mPool(NULL)
{
	mPool = PORT_NewArena(chunkSize);
	if(mPool == NULL) {
		THROW_ERROR;
	}
}

SecNssCoder::~SecNssCoder()
{
	if(mPool != NULL) {
		/*
		 * Note: we're asking for a memory zero here, but 
		 * PORT_FreeArena doesn't do that (yet).
		 */
		PORT_FreeArena(mPool, PR_TRUE);
		mPool = NULL;
	}
}

PRErrorCode	SecNssCoder::decode(
	const void				*src,		// BER-encoded source
	size_t				len,
	const SecAsn1Template 	*templ,	
	void					*dest)
{
	SECStatus prtn;
	
	assert(mPool != NULL);
	prtn = SEC_ASN1Decode(mPool, dest, templ, (const char *)src, len);
	if(prtn) {
		return PR_GetError();
	}
	else {
		return 0;
	}
}

PRErrorCode SecNssCoder::encodeItem(
	const void				*src,
	const SecAsn1Template 	*templ,	
	SECItem					&dest)
{
	assert(mPool != NULL);

	dest.Data = NULL;
	dest.Length = 0;
	
	SECItem *rtnItem = SEC_ASN1EncodeItem(mPool, &dest, src, templ);
	if(rtnItem == NULL) {
		return PR_GetError();
	}
	else {
		return 0;
	}
}

void *SecNssCoder::alloc(size_t len)
{
	assert(mPool != NULL);
	void *rtn = PORT_ArenaAlloc(mPool, len);
	if(rtn == NULL) {
		THROW_ERROR;
	}
	return rtn;
}

/* allocate space for num copies of specified type */
void *SecNssCoder::malloc_T(
                            size_t unit_bytesize,
                            size_t num_units)
{
    if (num_units>=SIZE_MAX/unit_bytesize) {
        THROW_ERROR;
        return NULL;
    }
    return alloc(unit_bytesize * num_units);
}


/* malloc item.Data, set item.Length */
void SecNssCoder::allocItem(
	SECItem					&item,
	size_t					len)
{
	item.Data = (uint8 *)alloc(len);
	item.Length = len;
}

/* malloc and copy */
void SecNssCoder::allocCopyItem(
	const void				*src,
	size_t					len,
	SECItem					&dest)
{
	allocItem(dest, len);
	memmove(dest.Data, src, len);
}

/*
 * This is pretty much a copy of SEC_ASN1EncodeItem, with a Allocator
 * malloc replacing the sec_asn1e_allocate_item to alloc the output data.
 */
PRErrorCode SecNssEncodeItem(
	const void				*src,
	const SecAsn1Template 	*templ,	
	Security::Allocator	&alloc,
	SECItem					&dest)
{
    unsigned long encoding_length = 0;
    SECStatus rv;

	dest.Data = NULL;
	dest.Length = 0;

    rv = SEC_ASN1Encode (src, templ,
			 sec_asn1e_encode_item_count, &encoding_length);
    if (rv != SECSuccess) {
		return PR_GetError();
	}

	/* replace this...
    dest = sec_asn1e_allocate_item (poolp, dest, encoding_length);
    if (dest == NULL)
	return NULL;
	... with this: */
	dest.Data = (uint8 *)alloc.malloc(encoding_length);
	dest.Length = 0;
	/* end replacement */
	
    rv = SEC_ASN1Encode (src, templ, sec_asn1e_encode_item_store, &dest);
    if (rv != SECSuccess) {
		return PR_GetError();
	}

    assert(encoding_length == dest.Length);
    return 0;
}

PRErrorCode SecNssEncodeItemOdata(
	const void				*src,
	const SecAsn1Template 	*templ,	
	CssmOwnedData			&odata)
{
	Allocator &alloc = odata.allocator;
	SECItem sitem;
	PRErrorCode prtn;
	
	prtn = SecNssEncodeItem(src, templ, alloc, sitem);
	if(prtn) {
		return prtn;
	}
	odata.set(sitem.Data, sitem.Length);
	return 0;
}

#endif /* TARGET_OS_MAC */
