/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
 * DecodedCrl.h - object representing a decoded cert in NSS form, with
 * extensions parsed and decoded (still in NSS format).
 *
 *
 * See DecodedItem.h for details on the care and feeding of this
 * module. 
 */

#ifndef	_DECODED_CRL_H_
#define _DECODED_CRL_H_

#include <Security/cssmtype.h>
#include <security_cdsa_utilities/cssmdata.h>

#include "DecodedItem.h"
#include <Security/X509Templates.h>

class DecodedCrl : /* for now public CertificateList, */ public DecodedItem
{
	NOCOPY(DecodedCrl)
public:
	/* construct empty CRL, no decoded extensions */
	DecodedCrl(
		AppleX509CLSession	&session);
	
	/* one-shot constructor, decoding from DER-encoded data */
	DecodedCrl(
		AppleX509CLSession	&session,
		const CssmData 		&encodedCrl);
		
	~DecodedCrl();
	
	/* decode CRLToSign and its extensions */
	void decodeCts(
		const CssmData	&encodedCTS);
		
	/* encode TBSCert and its extensions */
	void encodeExtensions();
	void encodeCts(
		CssmOwnedData	&encodedTbs);
		
	/***
	 *** field accessors (in CrlFields.cpp)
	 ***/
	
	/* 
	 * Obtain the index'th occurrence of field specified by fieldId.
	 * Format of the returned field depends on fieldId.
	 * Returns total number of fieldId fields in the cert if index is 0.
	 * Returns true if specified field was found, else returns false. 
	 */
	bool getCrlFieldData(
		const CssmOid		&fieldId,			// which field
		unsigned			index,				// which occurrence (0 = first)
		uint32				&numFields,			// RETURNED
		CssmOwnedData		&fieldValue);		// RETURNED

	/*
	 * Set the field specified by fieldId in TBS. 
	 * Note no index - individual field routines either append (for extensions)
	 * or throw if field already set (for all others) 
	 */
	void setCrlField(
		const CssmOid		&fieldId,		// which field
		const CssmData		&fieldValue);	

	/*
	 * Free the fieldId-specific data referred to by fieldValue.get().data().
	 */
	static void freeCrlFieldData(
		const CssmOid		&fieldId,
		CssmOwnedData		&fieldValue);

	void getAllParsedCrlFields(
		uint32 				&NumberOfFields,		// RETURNED
		CSSM_FIELD_PTR 		&CertFields);			// RETURNED

	static void describeFormat(
		Allocator 		&alloc,
		uint32 				&NumberOfFields,
		CSSM_OID_PTR 		&OidList);

	NSS_Crl	mCrl;
	
};

#endif	/* _DECODED_CRL_H_ */
