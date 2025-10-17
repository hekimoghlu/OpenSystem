/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
//
// Session_CSP.cpp - CSR-related session functions.
//

#include "AppleX509CLSession.h"
#include "DecodedCert.h"
#include "clNameUtils.h"
#include "clNssUtils.h"
#include "cldebugging.h"
#include "CSPAttacher.h"
#include "clNssUtils.h"
#include <Security/oidsattr.h>
#include <Security/oidscert.h>
#include <Security/cssmapple.h>
#include <Security/csrTemplates.h>
#include <Security/SecAsn1Templates.h>

/* 
 * Generate a DER-encoded CSR.
 */
void AppleX509CLSession::generateCsr(
	CSSM_CC_HANDLE 		CCHandle,
	const CSSM_APPLE_CL_CSR_REQUEST *csrReq,
	CSSM_DATA_PTR		&csrPtr)
{
	/*
	 * We use the full NSSCertRequest here; we encode the 
	 * NSSCertRequestInfo component separately to calculate
	 * its signature, then we encode the whole NSSCertRequest
	 * after dropping in the signature and SignatureAlgorithmIdentifier.
	 */ 
	NSSCertRequest certReq;
	NSSCertRequestInfo &reqInfo = certReq.reqInfo;
	PRErrorCode prtn;

	memset(&certReq, 0, sizeof(certReq));
	
	/* 
	 * Step 1: convert CSSM_APPLE_CL_CSR_REQUEST to CertificationRequestInfo.
	 * All allocs via local arena pool.
	 */
	SecNssCoder coder;
	ArenaAllocator alloc(coder);
	clIntToData(0, reqInfo.version, alloc);
	
	/* subject Name, required  */
	if(csrReq->subjectNameX509 == NULL) {
		CssmError::throwMe(CSSMERR_CL_INVALID_POINTER);
	}
	CL_cssmNameToNss(*csrReq->subjectNameX509, reqInfo.subject, coder);
	
	/* key --> CSSM_X509_SUBJECT_PUBLIC_KEY_INFO */
	CL_CSSMKeyToSubjPubKeyInfoNSS(*csrReq->subjectPublicKey, 
		reqInfo.subjectPublicKeyInfo, coder);

	/* attributes - see sm_x501if - we support one, CSSMOID_ChallengePassword,
 	 * as a printable string */
	if(csrReq->challengeString) {
		/* alloc a NULL_terminated array of NSS_Attribute pointers */
		reqInfo.attributes = (NSS_Attribute **)coder.alloc(2 * sizeof(NSS_Attribute *));
		reqInfo.attributes[1] = NULL;
		
		/* alloc one NSS_Attribute */
		reqInfo.attributes[0] = (NSS_Attribute *)coder.alloc(sizeof(NSS_Attribute));
		NSS_Attribute *attr = reqInfo.attributes[0];
		memset(attr, 0, sizeof(NSS_Attribute));
		
		 /* NULL_terminated array of attrValues */
		attr->attrValue = (CSSM_DATA **)coder.alloc(2 * sizeof(CSSM_DATA *));
		attr->attrValue[1] = NULL;
		
		/* one value - we're almost there */
		attr->attrValue[0] = (CSSM_DATA *)coder.alloc(sizeof(CSSM_DATA));
		
		/* attrType is an OID, temp, use static OID */
		attr->attrType = CSSMOID_ChallengePassword;

		/* one value, spec'd as AsnAny, we have to encode first. */		
		CSSM_DATA strData;
		strData.Data = (uint8 *)csrReq->challengeString;
		strData.Length = strlen(csrReq->challengeString);
		prtn = coder.encodeItem(&strData, kSecAsn1PrintableStringTemplate,
			*attr->attrValue[0]);
		if(prtn) {
			clErrorLog("generateCsr: error encoding challengeString\n");
			CssmError::throwMe(CSSMERR_CL_MEMORY_ERROR);
		}
	}
	
	/*
	 * Step 2: DER-encode the NSSCertRequestInfo prior to signing.
	 */
	CSSM_DATA encReqInfo;
	prtn = coder.encodeItem(&reqInfo, kSecAsn1CertRequestInfoTemplate, encReqInfo);
	if(prtn) {
		clErrorLog("generateCsr: error encoding CertRequestInfo\n");
		CssmError::throwMe(CSSMERR_CL_MEMORY_ERROR);
	}
	
	/*
	 * Step 3: sign the encoded NSSCertRequestInfo.
	 */
	CssmAutoData sig(*this);
	CssmData &infoData = CssmData::overlay(encReqInfo);
	signData(CCHandle, infoData, sig);
	 
	/*
	 * Step 4: finish up NSSCertRequest - signatureAlgorithm, signature
	 */
	certReq.signatureAlgorithm.algorithm = csrReq->signatureOid;
	/* FIXME - for now assume NULL alg params */
	CL_nullAlgParams(certReq.signatureAlgorithm);
	certReq.signature.Data = (uint8 *)sig.data();
	certReq.signature.Length = sig.length() * 8;
	
	/* 
	 * Step 5: DER-encode the finished NSSCertRequest into app space.
	 */
	CssmAutoData encCsr(*this);
	prtn = SecNssEncodeItemOdata(&certReq, kSecAsn1CertRequestTemplate, encCsr);
	if(prtn) {
		clErrorLog("generateCsr: error encoding CertRequestInfo\n");
		CssmError::throwMe(CSSMERR_CL_MEMORY_ERROR);
	}
	
	/* TBD - enc64 the result, when we have this much working */
	csrPtr = (CSSM_DATA_PTR)malloc(sizeof(CSSM_DATA));
	csrPtr->Data = (uint8 *)encCsr.data();
	csrPtr->Length = encCsr.length();
	encCsr.release();
}

/*
 * Verify CSR with its own public key. 
 */
void AppleX509CLSession::verifyCsr(
	const CSSM_DATA		*csrPtr)
{
	/*
	 * 1. Extract the public key from the CSR. We do this by decoding
	 *    the whole thing and getting a CSSM_KEY from the 
	 *    SubjectPublicKeyInfo.
	 */
	NSSCertRequest certReq;
	SecNssCoder coder;
	PRErrorCode prtn;
	
	memset(&certReq, 0, sizeof(certReq));
	prtn = coder.decodeItem(*csrPtr, kSecAsn1CertRequestTemplate, &certReq);
	if(prtn) {
		CssmError::throwMe(CSSMERR_CL_INVALID_DATA);
	}
	
	NSSCertRequestInfo &reqInfo = certReq.reqInfo;
	CSSM_KEY_PTR cssmKey = CL_extractCSSMKeyNSS(reqInfo.subjectPublicKeyInfo, 
		*this,		// alloc
		NULL);		// no DecodedCert

	/*
	 * 2. Obtain signature algorithm and parameters. 
	 */
	CSSM_X509_ALGORITHM_IDENTIFIER sigAlgId = certReq.signatureAlgorithm;
	CSSM_ALGORITHMS vfyAlg = CL_oidToAlg(sigAlgId.algorithm);
			
	/* 
	 * Handle CSSMOID_ECDSA_WithSpecified, which requires additional
	 * decode to get the digest algorithm.
	 */
	if(vfyAlg == CSSM_ALGID_ECDSA_SPECIFIED) {
		vfyAlg = CL_nssDecodeECDSASigAlgParams(sigAlgId.parameters, coder);
	}
	
	/*
	 * 3. Extract the raw bits to be verified and the signature. We 
	 *    decode the CSR as a CertificationRequestSigned for this, which 
	 *    avoids the decode of the CertificationRequestInfo.
	 */
	NSS_SignedCertRequest certReqSigned;
	memset(&certReqSigned, 0, sizeof(certReqSigned));
	prtn = coder.decodeItem(*csrPtr, kSecAsn1SignedCertRequestTemplate, &certReqSigned);
	if(prtn) {
		CssmError::throwMe(CSSMERR_CL_INVALID_DATA);
	}

	CSSM_DATA sigBytes = certReqSigned.signature;
	sigBytes.Length = (sigBytes.Length + 7 ) / 8;
	CssmData &sigCdata = CssmData::overlay(sigBytes);
	CssmData &toVerify = CssmData::overlay(certReqSigned.certRequestBlob);
	
	/*
	 * 4. Attach to CSP, cook up signature context, verify signature.
	 */
	CSSM_CSP_HANDLE cspHand = getGlobalCspHand(true);
	CSSM_RETURN crtn;
	CSSM_CC_HANDLE ccHand;
	crtn = CSSM_CSP_CreateSignatureContext(cspHand,
		vfyAlg,
		NULL,			// Access Creds
		cssmKey,
		&ccHand);
	if(crtn) {
		CssmError::throwMe(crtn);
	}
	verifyData(ccHand, toVerify, sigCdata);
	CL_freeCSSMKey(cssmKey, *this);
}

