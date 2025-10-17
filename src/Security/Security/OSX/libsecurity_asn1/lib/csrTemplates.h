/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#ifndef	_NSS_CSR_TEMPLATES_H_
#define _NSS_CSR_TEMPLATES_H_

#include <Security/X509Templates.h>
#include <Security/keyTemplates.h>	/* for NSS_Attribute */

#ifdef  __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/*
 * ASN class : CertificationRequestInfo
 * C struct  : NSSCertRequestInfo
 */
typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER {
	SecAsn1Item							version;
    NSS_Name 							subject;
    SecAsn1PubKeyInfo 	subjectPublicKeyInfo;
	NSS_Attribute						**attributes;
} NSSCertRequestInfo DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

extern const SecAsn1Template kSecAsn1CertRequestInfoTemplate[] SEC_ASN1_API_DEPRECATED;

/* 
 * ASN class : CertificationRequest
 * C struct  : NSSCertRequest
 */
typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER {
	NSSCertRequestInfo				reqInfo;
    SecAsn1AlgId 	signatureAlgorithm;
    SecAsn1Item 						signature;// BIT STRING, length in bits	
} NSSCertRequest DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

extern const SecAsn1Template kSecAsn1CertRequestTemplate[] SEC_ASN1_API_DEPRECATED;

/*
 * This is what we use use to avoid unnecessary setup and teardown of 
 * a full NSSCertRequest when signing and verifying.
 */
typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER {
	SecAsn1Item						certRequestBlob;	// encoded, ASN_ANY
    SecAsn1AlgId 	signatureAlgorithm;
    SecAsn1Item 						signature;// BIT STRING, length in bits	
} NSS_SignedCertRequest DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

extern const SecAsn1Template kSecAsn1SignedCertRequestTemplate[] SEC_ASN1_API_DEPRECATED;

#pragma clang diagnostic pop

#ifdef  __cplusplus
}
#endif

#endif	/* _NSS_CSR_TEMPLATES_H_ */
