/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
 * DH_utils.h
 */
#ifndef	_DH_UTILS_H_
#define _DH_UTILS_H_

#include <openssl/dh_legacy.h>
#include <AppleCSPSession.h>
#include <security_cdsa_utilities/context.h>

#ifdef	__cplusplus
extern "C" {
#endif

void throwDh(
	const char *op);
	
/* 
 * Given a Context:
 * -- obtain CSSM key (there must only be one)
 * -- validate keyClass - MUST be private! (DH public keys are never found
 *    in contexts.)
 * -- validate keyUsage
 * -- convert to DH *, allocating the DH key if necessary
 */
DH *contextToDhKey(
	const Context 		&context,
	AppleCSPSession	 	&session,
	CSSM_ATTRIBUTE_TYPE	attr,		  // CSSM_ATTRIBUTE_KEY for normal private key
									  // CSSM_ATTRIBUTE_PUBLIC_KEY for public key
	CSSM_KEYCLASS		keyClass,	  // CSSM_KEYCLASS_{PUBLIC,PRIVATE}_KEY	
	CSSM_KEYUSE			usage,		  // CSSM_KEYUSE_ENCRYPT, 
									  //    CSSM_KEYUSE_SIGN, etc.
	bool				&mallocdKey); // RETURNED

/* 
 * Convert a CssmKey to an DH * key. May result in the creation of a new
 * DH (when cssmKey is a raw key); allocdKey is true in that case
 * in which case the caller generally has to free the allocd key).
 */
DH *cssmKeyToDh(
	const CssmKey	&cssmKey,
	AppleCSPSession	&session,
	bool			&allocdKey);	// RETURNED

/* 
 * Convert a raw CssmKey to a newly alloc'd DH *.
 */
DH *rawCssmKeyToDh(
	const CssmKey	&cssmKey);


#ifdef	__cplusplus
}
#endif

#endif	/*_DH_UTILS_H_ */
