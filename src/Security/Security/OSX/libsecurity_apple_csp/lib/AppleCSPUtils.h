/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
// AppleCSPUtils.h - CSP-wide utility functions
//

#ifndef	_H_APPLE_CSP_UTILS
#define _H_APPLE_CSP_UTILS

#include "cspdebugging.h"
#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>
#include <security_cdsa_utilities/context.h>

#ifdef	__cplusplus
extern "C" {
#endif

/* Key type */
typedef enum {
	CKT_Session,
	CKT_Private,
	CKT_Public
} cspKeyType;

/* Key storage type returned from cspParseKeyAttr() */
typedef enum {
	 CKS_Ref,
	 CKS_Data,
	 CKS_None
} cspKeyStorage;

#define KEY_ATTR_RETURN_MASK	(CSSM_KEYATTR_RETURN_DATA |		\
								 CSSM_KEYATTR_RETURN_REF  |		\
								 CSSM_KEYATTR_RETURN_NONE)

/*
 * Validate key attribute bits per specified key type.
 *
 * Used to check requested key attributes for new keys and for validating
 * incoming existing keys. For checking key attributes for new keys,
 * assumes that KEYATTR_RETURN_xxx bits have been checked elsewhere
 * and stripped off before coming here.
 */
void cspValidateKeyAttr(
	cspKeyType 	keyType,
	uint32 		keyAttr);

/*
 * Perform check of incoming key attribute bits for a given
 * key type, and return a malKeyStorage value.
 *
 * Called from any routine which generates a new key. This specifically
 * excludes WrapKey().
 */
cspKeyStorage cspParseKeyAttr(
	cspKeyType 	keyType,
	uint32 		keyAttr);
	
/*
 * Validate key usage bits for specified key type.
 */
void cspValidateKeyUsageBits (
	cspKeyType	keyType,
	uint32		keyUsage);

/*
 * Validate existing key's usage bits against intended use.
 */
void cspValidateIntendedKeyUsage(
	const CSSM_KEYHEADER	*hdr,
	CSSM_KEYUSE				intendedUsage);

/*
 * Set up a key header.
 */
void setKeyHeader(
	CSSM_KEYHEADER &hdr,
	const Guid &myGuid,
	CSSM_ALGORITHMS alg, 
	CSSM_KEYCLASS keyClass,
	CSSM_KEYATTR_FLAGS attrs, 
	CSSM_KEYUSE use);

/*
 * Ensure that indicated CssmData can handle 'length' bytes 
 * of data. Malloc the Data ptr if necessary.
 */
void setUpCssmData(
	CssmData			&data,
	size_t				length,
	Allocator		&allocator);

void setUpData(
	CSSM_DATA			&data,
	size_t				length,
	Allocator		&allocator);
	
void freeCssmData(
	CssmData			&data, 
	Allocator		&allocator);
	
void freeData(
	CSSM_DATA			*data, 
	Allocator		&allocator,
	bool				freeStruct);		// free the CSSM_DATA itself

/*
 * Copy source to destination, mallocing destination if necessary.
 */
void copyCssmData(
	const CssmData		&src,
	CssmData			&dst,
	Allocator		&allocator);

void copyData(
	const CSSM_DATA		&src,
	CSSM_DATA			&dst,
	Allocator		&allocator);

/*
 * Compare two CSSM_DATAs, return CSSM_TRUE if identical.
 */
CSSM_BOOL cspCompareCssmData(
	const CSSM_DATA 	*data1,
	const CSSM_DATA 	*data2);

/*
 * This takes care of mallocing the and KeyLabel field. 
 */
void copyCssmHeader(
	const CssmKey::Header	&src,
	CssmKey::Header			&dst,
	Allocator			&allocator);
	
/*
 * Given a wrapped key, infer its raw format. 
 * This is a real kludge; it only works as long as each {algorithm, keyClass}
 * maps to exactly one format.  
 */
CSSM_KEYBLOB_FORMAT inferFormat(
	const CssmKey	&wrappedKey);

/*
 * Given a key and a Context, obtain the optional associated 
 * CSSM_ATTRIBUTE_{PUBLIC,PRIVATE,SYMMETRIC}_KEY_FORMAT attribute as a 
 * CSSM_KEYBLOB_FORMAT.
 */
CSSM_KEYBLOB_FORMAT requestedKeyFormat(
	const Context 	&context,
	const CssmKey	&key,
    CSSM_KEYBLOB_FORMAT defaultFormat = CSSM_KEYBLOB_RAW_FORMAT_NONE);

/* stateless function to calculate SHA-1 hash of a blob */

#define SHA1_DIGEST_SIZE	20
void cspGenSha1Hash(
	const void 		*inData,
	size_t			inDataLen,
	void			*out);			// caller mallocs, digest goes here

void cspVerifyKeyTimes(
	const CSSM_KEYHEADER &hdr);

#ifdef	__cplusplus
}
#endif

#endif	//  _H_APPLE_CSP_UTILS
