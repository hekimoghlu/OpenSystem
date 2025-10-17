/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
 * DecodedCert.h - object representing an NSS-decoded cert, with extensions
 * parsed and decoded (still in NSS format).
 *
 * Copyright (c) 2000,2011,2014 Apple Inc. 
 *
 * See DecodedItem.h for details on the care and feeding of this
 * module. 
 */

#ifndef	_DECODED_CERT_H_
#define _DECODED_CERT_H_

#include <Security/cssmtype.h>
#include <security_cdsa_utilities/cssmdata.h>

#include "DecodedItem.h"
#include <Security/X509Templates.h>
#include <security_asn1/SecNssCoder.h>

class DecodedCert : public DecodedItem
{
	NOCOPY(DecodedCert)
public:
	/* construct empty cert, no decoded extensions */
	DecodedCert(
		AppleX509CLSession	&session);
	
	/* one-shot constructor, decoding from DER-encoded data */
	DecodedCert(
		AppleX509CLSession	&session,
		const CssmData 		&encodedCert);
		
	~DecodedCert();
	
	void encodeExtensions();
	
	/* decode TBSCert and its extensions */
	void decodeTbs(
		const CssmData	&encodedTbs);
		
	/* encode TBSCert and its extensions */
	void encodeTbs(
		CssmOwnedData	&encodedTbs);
		
	/***
	 *** field accessors (in CertFields.cpp)
	 ***/
	
	/* 
	 * Obtain the index'th occurrence of field specified by fieldId.
	 * Format of the returned field depends on fieldId.
	 * Returns total number of fieldId fields in the cert if index is 0.
	 * Returns true if specified field was found, else returns false. 
	 */
	bool getCertFieldData(
		const CssmOid		&fieldId,			// which field
		unsigned			index,				// which occurrence (0 = first)
		uint32				&numFields,			// RETURNED
		CssmOwnedData		&fieldValue);		// RETURNED

	/*
	 * Set the field specified by fieldId in TBS. 
	 * Note no index - individual field routines either append (for extensions)
	 * or throw if field already set (for all others) 
	 */
	void setCertField(
		const CssmOid		&fieldId,		// which field
		const CssmData		&fieldValue);	

	/*
	 * Free the fieldId-specific data referred to by fieldValue.get().data().
	 */
	static void freeCertFieldData(
		const CssmOid		&fieldId,
		CssmOwnedData		&fieldValue);

	void getAllParsedCertFields(
		uint32 				&NumberOfFields,		// RETURNED
		CSSM_FIELD_PTR 		&CertFields);			// RETURNED

	static void describeFormat(
		Allocator 		&alloc,
		uint32 				&NumberOfFields,
		CSSM_OID_PTR 		&OidList);

	/*
  	 * Obtain a CSSM_KEY from a decoded cert, inferring as much as we can
	 * from required fields (subjectPublicKeyInfo) and extensions (for 
	 * KeyUse).
	 */
	CSSM_KEY_PTR extractCSSMKey(
		Allocator		&alloc) const;

	CSSM_KEYUSE inferKeyUsage() const;
	
	NSS_Certificate			mCert;
};

#endif	/* _DECODED_CERT_H_ */
