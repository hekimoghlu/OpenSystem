/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
#ifndef	_SEC_NSS_CODER_H_
#define _SEC_NSS_CODER_H_

#include <security_asn1/plarenas.h>
#include <security_asn1/prerror.h>
#include <Security/secasn1t.h>
#include <security_asn1/seccomon.h>
#include <security_utilities/alloc.h>
#include <security_cdsa_utilities/cssmdata.h>

/* 
 * Default chunk size for new arena pool.
 * FIXME: analyze & measure different defaults here. I'm pretty sure
 * that only performance - not correct behavior - is affected by
 * an arena pool's chunk size.
 */
#define SNC_CHUNKSIZE_DEF		1024		

class SecNssCoder
{
public:
	SecNssCoder(
		PRUint32 chunkSize = SNC_CHUNKSIZE_DEF);
	~SecNssCoder();
	
	/*
	 * BER decode an untyped item per the specified
	 * template array. The result is allocated 
	 * by this object's PLArenaPool and is freed when 
	 * this object is deleted.
	 *
	 * The dest pointer is a template-specific struct allocated
	 * by the caller and must be zeroed by the caller. 
	 *
	 * This does not throw any exceptions; error status 
	 * (obtained from PR_GetError() is returned. 
	 */
	PRErrorCode	decode(
		const void				*src,		// BER-encoded source
		size_t				len,
		const SecAsn1Template 	*templ,	
		void					*dest);
		
	/* convenience routine, decode from an SECItem */
	PRErrorCode	decodeItem(
		const SECItem			&item,		// BER-encoded source
		const SecAsn1Template 	*templ,	
		void					*dest)
		{
			return decode(item.Data, item.Length, templ, dest);
		}
		
	
	/*
	 * BER-encode. This object's arena pool retains a copy of 
	 * the encoded data.
	 *
	 * The src pointer is a template-specific struct.
	 * 
	 * This does not throw any exceptions; error status 
	 * (obtained from PR_GetError() is returned. 
	 */
	PRErrorCode encodeItem(
		const void				*src,
		const SecAsn1Template 	*templ,	
		SECItem					&dest);
		
	/*
	 * Some alloc-related methods which come in handy when using
	 * this class. All memory is allocated using this object's 
	 * arena pool. Caller never has to free it. Used for
	 * temp allocs of memory which only needs a scope which is the
	 * same as this object. 
	 *
	 * These throw a CssmError in the highly unlikely event of 
	 * a malloc failure.
	 */
	/*
	 * Don't use the name 'malloc' or the static analyzer may
	 * falsely flag potential leaks. Per above, the caller never
	 * has to free memory allocated by this method.
	 */
	void *alloc(
		size_t					len);
		
	/* allocate space for num copies of specified type */
	template <class T> T *mallocn(unsigned num = 1) 
		{ return reinterpret_cast<T *>(malloc_T(sizeof(T),num)); }

	/* malloc item.Data, set item.Length */
	void allocItem(
		SECItem					&item,
		size_t					len);
		
	/* malloc and copy, various forms */
	void allocCopyItem(
		const void				*src,
		size_t					len,
		SECItem					&dest);
		
	void allocCopyItem(
		const SECItem			&src,
		SECItem					&dest)
			{ allocCopyItem(src.Data, src.Length, dest); }
			
	void allocCopyItem(
		const CssmData			&src,
		SECItem					&dest)
			{ allocCopyItem(src.data(), src.length(), dest); }
		
	PLArenaPool	*pool() const { return mPool;}
	
private:
	PLArenaPool		*mPool;

    void *malloc_T(size_t unit_bytesize,
                   size_t num_units);
};

/*
 * Stateless function to BER-encode directly into a Allocator's
 * space. The only persistent allocated memory is allocated by 
 * the Allocator.
 *
 * The src pointer is a template-specific struct.
 * 
 * This does not throw any exceptions; error status 
 * (obtained from PR_GetError() is returned. 
 */
PRErrorCode SecNssEncodeItem(
	const void				*src,
	const SecAsn1Template 	*templ,	
	Allocator			&alloc,
	SECItem					&dest);

/*
 * Same thing, using a CssmOwnedData.
 */
PRErrorCode SecNssEncodeItemOdata(
	const void				*src,
	const SecAsn1Template 	*templ,	
	CssmOwnedData			&odata);

#endif	/* _SEC_NSS_CODER_H_ */
