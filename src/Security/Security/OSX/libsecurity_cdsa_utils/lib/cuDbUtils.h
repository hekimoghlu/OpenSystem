/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
	File:		 cuDbUtils.h
	
	Description: CDSA DB access utilities

	Author:		 dmitch
*/

#ifndef	_CU_DB_UTILS_H_
#define _CU_DB_UTILS_H_

#include <Security/cssm.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Add a certificate to an open DLDB.
 */
CSSM_RETURN cuAddCertToDb(
	CSSM_DL_DB_HANDLE	dlDbHand,
	const CSSM_DATA		*cert,
	CSSM_CERT_TYPE		certType,
	CSSM_CERT_ENCODING	certEncoding,
	const char			*printName,			// C string
	const CSSM_DATA		*publicKeyHash); 	// ??

/*
 * Add a CRL to an open DL/DB.
 */
CSSM_RETURN cuAddCrlToDb(
	CSSM_DL_DB_HANDLE	dlDbHand,
	CSSM_CL_HANDLE		clHand,
	const CSSM_DATA		*crl,
	const CSSM_DATA		*URI);

/*
 * Search DB for all records of type CRL or cert, calling appropriate
 * parse/print routine for each record. 
 */ 
CSSM_RETURN cuDumpCrlsCerts(
	CSSM_DL_DB_HANDLE	dlDbHand,
	CSSM_CL_HANDLE		clHand,
	CSSM_BOOL			isCert,
	unsigned			&numItems,		// returned
	CSSM_BOOL			verbose);

#ifdef	__cplusplus
}
#endif

#endif	/* _CU_DB_UTILS_H_ */
