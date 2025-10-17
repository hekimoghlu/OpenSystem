/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
 * tpOcspCertVfy.h - OCSP cert verification routines
 */
 
#ifndef	_TP_OCSP_CERT_VFY_H_
#define _TP_OCSP_CERT_VFY_H_

#include "TPCertInfo.h"
#include "tpCrlVerify.h"
#include <security_asn1/SecNssCoder.h>
#include <security_ocspd/ocspResponse.h>

#ifdef __cplusplus

extern "C" {
#endif

/*
 * Verify an OCSP response in the form of a pre-decoded OCSPResponse. Does 
 * signature verification as well as cert chain verification. Sometimes we can
 * verify if we don't know the issuer; sometimes we can.
 */
typedef enum {
	ORS_Unknown,			// unable to verify one way or another
	ORS_Good,				// known to be good
	ORS_Bad					// known to be bad
} OcspRespStatus;

OcspRespStatus tpVerifyOcspResp(
	TPVerifyContext		&vfyCtx,
	SecNssCoder			&coder,
	TPCertInfo			*issuer,		// issuer of the related cert, may be issuer of 
										//   reply
	OCSPResponse		&ocspResp,
	CSSM_RETURN			&cssmErr);		// possible per-cert error 

#ifdef __cplusplus
}
#endif

#endif	/* _TP_OCSP_CERT_VFY_H_ */

