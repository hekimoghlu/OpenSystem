/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
	tpPolicies.h - TP module policy implementation
*/

#ifndef	_TP_POLICIES_H_
#define _TP_POLICIES_H_

#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>
#include <Security/cssmapple.h>
#include "TPCertInfo.h"

#ifdef __cplusplus
extern	"C" {
#endif /* __cplusplus */

/*
 * Enumerated certificate policies enforced by this module.
 */
typedef enum {
	kTPDefault,			/* no extension parsing, just sig and expiration */
	kTPx509Basic,		/* basic X.509/RFC3280 */
	kTPiSign,			/* (obsolete) Apple code signing */
	kTP_SSL,			/* SecureTransport/SSL */
	kCrlPolicy,			/* cert chain verification via CRL */
	kTP_SMIME,			/* S/MIME */
	kTP_EAP,
	kTP_SWUpdateSign,	/* Apple SW Update signing (was Apple Code Signing) */
	kTP_ResourceSign,	/* Apple Resource Signing */
	kTP_IPSec,			/* IPSEC */
	kTP_iChat,			/* iChat */
	kTP_PKINIT_Client,	/* PKINIT client cert */
	kTP_PKINIT_Server,	/* PKINIT server cert */
	kTP_CodeSigning,	/* new Apple Code Signing (Leopard/10.5) */
	kTP_PackageSigning,	/* Package Signing */
	kTP_MacAppStoreRec,	/* MacApp store receipt */
	kTP_AppleIDSharing,	/* AppleID Sharing */
	kTP_TimeStamping,	/* RFC3161 time stamping */
	kTP_PassbookSigning,	/* Passbook Signing */
	kTP_MobileStore,	/* Apple Mobile Store Signing */
	kTP_TestMobileStore,	/* Apple Test Mobile Store Signing */
	kTP_EscrowService,	/* Apple Escrow Service Signing */
	kTP_ProfileSigning,	/* Apple Configuration Profile Signing */
	kTP_QAProfileSigning,	/* Apple QA Configuration Profile Signing */
	kTP_PCSEscrowService,	/* Apple PCS Escrow Service Signing */
	kTP_ProvisioningProfileSigning, /* Apple OS X Provisioning Profile Signing */
} TPPolicy;

/*
 * Perform TP verification on a constructed (ordered) cert group.
 */
CSSM_RETURN tp_policyVerify(
	TPPolicy						policy,
	Allocator						&alloc,
	CSSM_CL_HANDLE					clHand,
	CSSM_CSP_HANDLE					cspHand,
	TPCertGroup 					*certGroup,
	CSSM_BOOL						verifiedToRoot,		// last cert is good root
	CSSM_BOOL						verifiedViaTrustSetting,// last cert has valid user trust
	CSSM_APPLE_TP_ACTION_FLAGS		actionFlags,
	const CSSM_DATA					*policyFieldData,	// optional
    void 							*policyControl);	// future use

/*
 * Obtain policy-specific User Trust parameters
 */
void tp_policyTrustSettingParams(
	TPPolicy				policy,
	const CSSM_DATA			*policyFieldData,		// optional
	/* returned values - not mallocd */
	const char				**policyStr,
	uint32					*policyStrLen,
	SecTrustSettingsKeyUsage	*keyUse);

#ifdef __cplusplus
}
#endif
#endif	/* _TP_POLICIES_H_ */
