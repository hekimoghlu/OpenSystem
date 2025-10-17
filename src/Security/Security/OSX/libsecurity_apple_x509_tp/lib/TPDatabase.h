/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
 * TPDatabase.h - TP's DL/DB access functions.
 *
 */
 
#ifndef	_TP_DATABASE_H_
#define _TP_DATABASE_H_

#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>
#include "TPCertInfo.h"

#ifdef	__cplusplus
extern "C" {
#endif

TPCertInfo *tpDbFindIssuerCert(
	Allocator				&alloc,
	CSSM_CL_HANDLE			clHand,
	CSSM_CSP_HANDLE			cspHand,
	const TPClItemInfo		*subjectItem,
	const CSSM_DL_DB_LIST	*dbList,
	const char 				*verifyTime,		// may be NULL
	bool					&partialIssuerKey,	// RETURNED
    TPCertInfo              *oldRoot);

/*
 * Search a list of DBs for a CRL from the specified issuer and (optional)
 * TPVerifyContext.verifyTime. 
 * Just a boolean return - we found it, or not. If we did, we return a
 * TPCrlInfo which has been verified with the specified TPVerifyContext.
 */
class TPCrlInfo;
class TPVerifyContext;

TPCrlInfo *tpDbFindIssuerCrl(
	TPVerifyContext		&vfyCtx,
	const CSSM_DATA		&issuer,
	TPCertInfo			&forCert);

#if WRITE_FETCHED_CRLS_TO_DB
/*
 * Store a CRL in a DLDB.
 */
CSSM_RETURN tpDbStoreCrl(
	TPCrlInfo			&crl,
	CSSM_DL_DB_HANDLE	&dlDb);

#endif	/* WRITE_FETCHED_CRLS_TO_DB */

#ifdef	__cplusplus
}
#endif

#endif	/* _TP_DATABASE_H_ */
