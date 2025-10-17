/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
/*!
	@header SecOCSPRequest
	The functions and data types in SecOCSPRequest implement ocsp request
    creation.
*/

#ifndef _SECURITY_SECOCSPREQUEST_H_
#define _SECURITY_SECOCSPREQUEST_H_

#include <CoreFoundation/CFData.h>
#include <libDER/libDER.h>
#include <Security/Security.h>

__BEGIN_DECLS

/*
 CertID          ::=     SEQUENCE {
     hashAlgorithm       AlgorithmIdentifier,
     issuerNameHash      OCTET STRING, -- Hash of Issuer's DN
     issuerKeyHash       OCTET STRING, -- Hash of Issuers public key
     serialNumber        CertificateSerialNumber }
 */
typedef struct {
    DERItem        hashAlgorithm;
    DERItem        issuerNameHash;
    DERItem        issuerKeyHash;
    DERItem        serialNumber;
} DER_OCSPCertID;

extern const DERSize DERNumOCSPCertIDItemSpecs;
extern const DERItemSpec DER_OCSPCertIDItemSpecs[];

/*!
	@typedef SecOCSPRequestRef
	@abstract Object used for ocsp response decoding.
*/
typedef struct __SecOCSPRequest *SecOCSPRequestRef;

struct __SecOCSPRequest {
    SecCertificateRef certificate;
    SecCertificateRef issuer;
    CFDataRef der;
    DERItem certIdHash;
    CFDataRef issuerNameDigest;
    CFDataRef issuerPubKeyDigest;
    CFDataRef serial;
};

/*!
	@function SecOCSPRequestCreate
	@abstract Returns a SecOCSPRequestRef from a BER encoded ocsp response.
	@param certificate The certificate for which we want a OCSP request created.
	@param issuer The parent of certificate.
	@result A SecOCSPRequestRef.
*/
SecOCSPRequestRef SecOCSPRequestCreate(SecCertificateRef certificate,
    SecCertificateRef issuer);

/*!
	@function SecOCSPRequestCopyDER
	@abstract Returns a DER encoded ocsp request.
	@param ocspRequest A SecOCSPRequestRef.
	@result DER encoded ocsp request.
*/
CFDataRef SecOCSPRequestGetDER(SecOCSPRequestRef ocspRequest);

/*!
	@function SecOCSPRequestFinalize
	@abstract Frees a SecOCSPRequestRef.
	@param ocspRequest A SecOCSPRequestRef.
	@note The passed in SecOCSPRequestRef is deallocated
*/
void SecOCSPRequestFinalize(SecOCSPRequestRef ocspRequest);

/* Testing Decoder */
SecOCSPRequestRef SecOCSPRequestCreateWithData(CFDataRef der_ocsp_request);
SecOCSPRequestRef SecOCSPRequestCreateWithDataStrict(CFDataRef der_ocsp_request);

__END_DECLS

#endif /* !_SECURITY_SECOCSPREQUEST_H_ */
