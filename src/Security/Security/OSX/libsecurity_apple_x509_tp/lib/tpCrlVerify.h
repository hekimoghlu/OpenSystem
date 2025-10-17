/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
 * tpCrlVerify.h - routines to verify CRLs and to verify certs against CRLs.
 *
 */
 
#ifndef	_TP_CRL_VERIFY_H_
#define _TP_CRL_VERIFY_H_

#include <Security/cssmtype.h>
#include <security_utilities/alloc.h>
#include <Security/cssmapple.h>
#include <Security/cssmapplePriv.h>

class TPCertInfo;
class TPCertGroup;
class TPCrlInfo;
class TPCrlGroup;

/*
 * Enumerated CRL policies enforced by this module.
 */
typedef enum {
	kRevokeNone,			/* no revocation checking */
	kRevokeCrlBasic,
	kRevokeOcsp	
} TPRevocationPolicy;

/* Module-specific default policy */
#define TP_CRL_POLICY_DEFAULT	kRevokeNone

/*
 * Various parameters widely used in any operation involving CRL and 
 * OCSP verification. Most fields are optional.
 */
class TPVerifyContext {
	NOCOPY(TPVerifyContext)
public:
	TPVerifyContext(
		Allocator			&_alloc,
		CSSM_CL_HANDLE		_clHand,
		CSSM_CSP_HANDLE		_cspHand,
		CSSM_TIMESTRING		_verifyTime,
		uint32				_numAnchorCerts,
		const CSSM_DATA		*_anchorCerts,
		TPCertGroup			*_signerCerts,
		TPCrlGroup			*_inputCrls,
		TPCertGroup			&_gatheredCerts,
		CSSM_DL_DB_LIST_PTR	_dbList,
		TPRevocationPolicy	_policy,
		CSSM_APPLE_TP_ACTION_FLAGS	_actionFlags,
		CSSM_APPLE_TP_CRL_OPTIONS	*_crlOpts,
		CSSM_APPLE_TP_OCSP_OPTIONS	*_ocspOpts,
		const CSSM_OID		*_policyOid,
		const char			*_policyStr,
		uint32				_policyStrLen,
		CSSM_KEYUSE			_keyUse)
			: alloc(_alloc),
				clHand(_clHand),
				cspHand(_cspHand),
				verifyTime(_verifyTime),
				numAnchorCerts(_numAnchorCerts),
				anchorCerts(_anchorCerts),
				signerCerts(_signerCerts),
				inputCrls(_inputCrls),
				gatheredCerts(_gatheredCerts),
				dbList(_dbList),
				policy(_policy),
				actionFlags(_actionFlags),
				crlOpts(_crlOpts),
				ocspOpts(_ocspOpts),
				policyOid(_policyOid),
				policyStr(_policyStr),
				policyStrLen(_policyStrLen),
				keyUse(_keyUse)
					{ }
	
	~TPVerifyContext() { }
	
	Allocator						&alloc;
	CSSM_CL_HANDLE					clHand;
	CSSM_CSP_HANDLE					cspHand;
	
	/* 
	 * NULL means "verify for this momemt", otherwise indicates 
	 * time at which an entity is to be verified.
	 */
    CSSM_TIMESTRING 				verifyTime;
	
	/* trusted anchors */
	/* FIXME - maybe this should be a TPCertGroup */
    uint32 							numAnchorCerts;
	const CSSM_DATA					*anchorCerts;
	
	/* 
	 * Intermediate signing certs. Always present.
	 * This could come from the raw cert group to be verified
	 * in CertGroupVerify(), or the explicit SignerCertGroup in
	 * CrlVerify(). IN both cases the cert group owns the certs and 
	 * eventually frees them. These certs have not been verified in any 
	 * way other than to ensure that they parse and have been cached
	 * by the CL.
	 */
	TPCertGroup						*signerCerts;

	/* Raw CRLs provided by caller, state unknown, optional */
	TPCrlGroup						*inputCrls;
	
	/*
	 * Other certificates gathered during the course of this operation,
	 * currently consisting of certs fetched from DBs and from the net.
	 * This is currently set to AppleTPSession::CertGroupVerify's
	 * certsToBeFreed, to include certs fetched from the net (a
	 * significant optimization) and from DLDB (a side effect, also
	 * a slight optimization).
	 */
	TPCertGroup						&gatheredCerts;
	
	/* can contain certs and/or CRLs */
    CSSM_DL_DB_LIST_PTR 			dbList;
	
	TPRevocationPolicy				policy;
	CSSM_APPLE_TP_ACTION_FLAGS		actionFlags;
	
	/* one of these valid, depends on policy */
	const CSSM_APPLE_TP_CRL_OPTIONS	*crlOpts;
	const CSSM_APPLE_TP_OCSP_OPTIONS *ocspOpts;
	
	/* optional user trust parameters */
	const CSSM_OID					*policyOid;
	const char						*policyStr;
	uint32							policyStrLen;
	CSSM_KEYUSE						keyUse;
};

extern "C" {

/* CRL - specific */
CSSM_RETURN tpVerifyCertGroupWithCrls(
	TPVerifyContext					&tpVerifyContext,
	TPCertGroup 					&certGroup);		// to be verified 
	
/* general purpose, switch to policy-specific code based on TPVerifyContext.policy */
CSSM_RETURN tpRevocationPolicyVerify(
	TPVerifyContext					&tpVerifyContext,
	TPCertGroup 					&certGroup);		// to be verified 

}

#endif	/* _TP_CRL_VERIFY_H_ */
