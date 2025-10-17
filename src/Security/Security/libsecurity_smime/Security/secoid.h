/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#ifndef _SECOID_H_
#define _SECOID_H_
/*
 * secoid.h - public data structures and prototypes for ASN.1 OID functions
 */

#include <Security/secasn1t.h>
#include <security_asn1/plarenas.h>
#include <security_asn1/seccomon.h>

#include "libsecurity_smime/lib/secoidt.h"

SEC_BEGIN_PROTOS

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern const SecAsn1Template SECOID_AlgorithmIDTemplate[];

/* This functions simply returns the address of the above-declared template. */
SEC_ASN1_CHOOSER_DECLARE(SECOID_AlgorithmIDTemplate)

/*
 * OID handling routines
 */
extern SECOidData* SECOID_FindOID(const SecAsn1Item* oid);
extern SECOidTag SECOID_FindOIDTag(const SecAsn1Item* oid);
extern SECOidData* SECOID_FindOIDByTag(SECOidTag tagnum);
extern SECOidData* SECOID_FindOIDByCssmAlgorithm(SecAsn1AlgId cssmAlgorithm);

/****************************************/
/*
** Algorithm id handling operations
*/

/*
** Fill in an algorithm-ID object given a tag and some parameters.
** 	"aid" where the DER encoded algorithm info is stored (memory
**	   is allocated)
**	"tag" the tag defining the algorithm (SEC_OID_*)
**	"params" if not NULL, the parameters to go with the algorithm
*/
extern SECStatus
SECOID_SetAlgorithmID(PRArenaPool* arena, SECAlgorithmID* aid, SECOidTag tag, const SecAsn1Item* params);

/*
** Copy the "src" object to "dest". Memory is allocated in "dest" for
** each of the appropriate sub-objects. Memory in "dest" is not freed
** before memory is allocated (use SECOID_DestroyAlgorithmID(dest, PR_FALSE)
** to do that).
*/
extern SECStatus
SECOID_CopyAlgorithmID(PRArenaPool* arena, SECAlgorithmID* dest, const SECAlgorithmID* src);

/*
** Get the SEC_OID_* tag for the given algorithm-id object.
*/
extern SECOidTag SECOID_GetAlgorithmTag(const SECAlgorithmID* aid);

/*
** Destroy an algorithm-id object.
**	"aid" the certificate-request to destroy
**	"freeit" if PR_TRUE then free the object as well as its sub-objects
*/
extern void SECOID_DestroyAlgorithmID(SECAlgorithmID* aid, Boolean freeit);

/*
** Compare two algorithm-id objects, returning the difference between
** them.
*/
extern SECComparison SECOID_CompareAlgorithmID(const SECAlgorithmID* a, const SECAlgorithmID* b);

extern Boolean SECOID_KnownCertExtenOID(const SecAsn1Item* extenOid);

/* Given a SEC_OID_* tag, return a string describing it.
 */
extern const char* SECOID_FindOIDTagDescription(SECOidTag tagnum);

#if 0
/*
 * free up the oid data structures.
 */
extern SECStatus SECOID_Shutdown(void);
#endif

#pragma clang diagnostic pop

SEC_END_PROTOS

#endif /* _SECOID_H_ */
