/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
#ifndef _OIDSOCSP_H_
#define _OIDSOCSP_H_  1

#include <Security/SecAsn1Types.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const SecAsn1Oid
	/* OCSP */
	OID_PKIX_OCSP SEC_ASN1_API_DEPRECATED,
	OID_PKIX_OCSP_BASIC SEC_ASN1_API_DEPRECATED,
	OID_PKIX_OCSP_NONCE SEC_ASN1_API_DEPRECATED,
	OID_PKIX_OCSP_CRL SEC_ASN1_API_DEPRECATED,
	OID_PKIX_OCSP_RESPONSE SEC_ASN1_API_DEPRECATED,
	OID_PKIX_OCSP_NOCHECK SEC_ASN1_API_DEPRECATED,
	OID_PKIX_OCSP_ARCHIVE_CUTOFF SEC_ASN1_API_DEPRECATED,
    OID_PKIX_OCSP_SERVICE_LOCATOR SEC_ASN1_API_DEPRECATED,
    OID_GOOGLE_OCSP_SCT SEC_ASN1_API_DEPRECATED;

#ifdef __cplusplus
}
#endif

#endif /* _OIDSOCSP_H_ */
