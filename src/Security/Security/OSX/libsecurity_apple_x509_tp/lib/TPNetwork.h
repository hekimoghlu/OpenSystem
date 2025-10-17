/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
 * TPNetwork.h - LDAP (and eventually) other network tools 
 *
 */
 
#ifndef	_TP_NETWORK_H_
#define _TP_NETWORK_H_

#include <Security/cssmtype.h>
#include "TPCertInfo.h"
#include "TPCrlInfo.h"

extern "C" {

/*
 * Fetch CRL(s) for specified cert if the cert has a cRlDistributionPoint
 * extension. If a non-NULL CRL is returned, it has passed verification
 * with specified TPVerifyContext.
 * The common, trivial failure of "no URI in a cRlDistributionPoint 
 * extension" is indicated by CSSMERR_APPLETP_CRL_NOT_FOUND.
 */
extern CSSM_RETURN tpFetchCrlFromNet(
	TPCertInfo 			&cert,
	TPVerifyContext		&verifyContext,
	TPCrlInfo			*&crl);				// RETURNED

/*
 * Fetch issuer cert of specified cert if the cert has an issuerAltName
 * with a URI. If non-NULL cert is returned, it has passed subject/issuer
 * name comparison and signature verification with target cert.
 * The common, trivial failure of "no URI in an issuerAltName 
 * extension" is indicated by CSSMERR_TP_CERTGROUP_INCOMPLETE.
 * A CSSMERR_CSP_APPLE_PUBLIC_KEY_INCOMPLETE return indicates that
 * subsequent signature verification is needed. 
 */
extern CSSM_RETURN tpFetchIssuerFromNet(
	TPCertInfo			&subject,
	CSSM_CL_HANDLE		clHand,
	CSSM_CSP_HANDLE		cspHand,
	const char			*verifyTime,
	TPCertInfo			*&issuer);			// RETURNED
	
}

#endif	/* TP_NETWORK_H_ */
