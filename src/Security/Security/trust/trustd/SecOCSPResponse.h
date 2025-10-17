/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
	@header SecOCSPResponse
	The functions and data types in SecOCSPResponse implement ocsp response
    decoding and verification.
*/

#ifndef _SECURITY_SECOCSPRESPONSE_H_
#define _SECURITY_SECOCSPRESPONSE_H_

#include <Security/SecAsn1Coder.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFDate.h>
#include "trust/trustd/SecOCSPRequest.h"
#include <security_asn1/ocspTemplates.h>
#include <libDER/DER_Decode.h>
#include <libDER/DER_Keys.h>

__BEGIN_DECLS

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// rdar://56649059 (SecOCSPRequest and SecOCSPResponse should use libDER instead of libASN1)

/*
 OCSPResponseStatus ::= ENUMERATED {
     successful            (0), -- Response has valid confirmations
     malformedRequest      (1), -- Illegal confirmation request
     internalError         (2), -- Internal error in issuer
     tryLater              (3), -- Try again later
                                -- (4) is not used
     sigRequired           (5), -- Must sign the request
     unauthorized          (6)  -- Request unauthorized
 }
*/
typedef CF_ENUM(uint8_t, OCSPResponseStatus) {
    OCSPResponseStatusSuccessful = 0,
    OCSPResponseStatusMalformedRequest = 1,
    OCSPResponseStatusTnternalError = 2,
    OCSPResponseStatusTryLater = 3,
    OCSPResponseStatusSigRequired = 5,
    OCSPResponseStatusUnauthorized = 6,
};

typedef struct {
    DERItem responseData;
    DERItem signatureAlgorithm;
    DERItem signature;
    DERItem certs;
} DERBasicOCSPResponse;

typedef struct {
    DERItem version;
    DERItem responderId;
    DERItem producedAt;
    DERItem responses;
    DERItem extensions;
} DER_OCSPResponseData;

enum {
    kSecRevocationReasonUnrevoked               = -2,
    kSecRevocationReasonUndetermined            = -1,
    kSecRevocationReasonUnspecified             = 0,
    kSecRevocationReasonKeyCompromise           = 1,
    kSecRevocationReasonCACompromise            = 2,
    kSecRevocationReasonAffiliationChanged      = 3,
    kSecRevocationReasonSuperseded              = 4,
    kSecRevocationReasonCessationOfOperation    = 5,
    kSecRevocationReasonCertificateHold         = 6,
    /*         -- value 7 is not used */
    kSecRevocationReasonRemoveFromCRL           = 8,
    kSecRevocationReasonPrivilegeWithdrawn      = 9,
    kSecRevocationReasonAACompromise            = 10
};
typedef int32_t SecRevocationReason;

/*!
	@typedef SecOCSPResponseRef
	@abstract Object used for ocsp response decoding.
*/
typedef struct __SecOCSPResponse *SecOCSPResponseRef;

struct __SecOCSPResponse {
        CFDataRef data;
        OCSPResponseStatus responseStatus;
        CFAbsoluteTime producedAt;
        CFAbsoluteTime latestNextUpdate;
        CFAbsoluteTime expireTime;
        DERBasicOCSPResponse basicResponse;
        DER_OCSPResponseData responseData;
        DERDecodedInfo responderId;
        int64_t responseID;
};

typedef CF_ENUM(uint8_t, OCSPCertStatus) {
    OCSPCertStatusGood = 0,
    OCSPCertStatusRevoked = 1,
    OCSPCertStatusUnknown = 2,
};

typedef struct __SecOCSPSingleResponse *SecOCSPSingleResponseRef;

struct __SecOCSPSingleResponse {
    OCSPCertStatus certStatus;
    CFAbsoluteTime thisUpdate;
    CFAbsoluteTime nextUpdate;		/* may be NULL_TIME */
    CFAbsoluteTime revokedTime;		/* != NULL_TIME for certStatus == CS_Revoked */
    SecRevocationReason crlReason;
    CFArrayRef scts;                /* This is parsed from an extension */
};

#pragma clang diagnostic pop

/*!
	@function SecOCSPResponseCreate
	@abstract Returns a SecOCSPResponseRef from a BER encoded ocsp response.
	@param ocspResponse The BER encoded ocsp response.
	@result A SecOCSPResponseRef.
*/
SecOCSPResponseRef SecOCSPResponseCreate(CFDataRef ocspResponse);

SecOCSPResponseRef SecOCSPResponseCreateWithID(CFDataRef ocspResponse, int64_t responseID);

int64_t SecOCSPResponseGetID(SecOCSPResponseRef ocspResponse);

/* Return true if response is still valid for the given age. */
bool SecOCSPResponseCalculateValidity(SecOCSPResponseRef this,
    CFTimeInterval maxAge, CFTimeInterval defaultTTL, CFAbsoluteTime verifyTime);

CFDataRef SecOCSPResponseGetData(SecOCSPResponseRef this);

OCSPResponseStatus SecOCSPGetResponseStatus(SecOCSPResponseRef ocspResponse);

CFAbsoluteTime SecOCSPResponseGetExpirationTime(SecOCSPResponseRef ocspResponse);

CFAbsoluteTime SecOCSPResponseProducedAt(SecOCSPResponseRef ocspResponse);

/*!
	@function SecOCSPResponseCopySigners
	@abstract Returns an array of signers.
	@param ocspResponse A SecOCSPResponseRef.
	@result The passed in SecOCSPResponseRef is deallocated
*/
CFArrayRef SecOCSPResponseCopySigners(SecOCSPResponseRef ocspResponse);

/*!
	@function SecOCSPResponseFinalize
	@abstract Frees a SecOCSPResponseRef.
	@param ocspResponse The BER encoded ocsp response.
*/
void SecOCSPResponseFinalize(SecOCSPResponseRef ocspResponse);

SecOCSPSingleResponseRef SecOCSPResponseCopySingleResponse(
    SecOCSPResponseRef ocspResponse, SecOCSPRequestRef request);

/* DefaultTTL is how long past the thisUpdate time we trust a response without a nextUpdate field.  */
bool SecOCSPSingleResponseCalculateValidity(SecOCSPSingleResponseRef this, CFAbsoluteTime defaultTTL, CFAbsoluteTime verifyTime);

/* Find the eventual SCTs from the single response extensions */
CFArrayRef SecOCSPSingleResponseCopySCTs(SecOCSPSingleResponseRef this);

void SecOCSPSingleResponseDestroy(SecOCSPSingleResponseRef this);

bool SecOCSPResponseForSingleResponse(SecOCSPResponseRef response, DERReturn (^operation)(SecOCSPSingleResponseRef singleResponse, DER_OCSPCertID *certId, DERAlgorithmId *hashAlgorithm, bool *stop));

/* Returns the SecCertificateRef whose leaf signed this ocspResponse if
   we can find one and NULL if we can't find a valid signer. The issuerPath
   contains the cert chain from the anchor to the certificate that issued the
   leaf certificate for which this ocspResponse is supposed to be valid. */
SecCertificateRef SecOCSPResponseCopySigner(SecOCSPResponseRef this,
    SecCertificateRef issuerPath);

bool SecOCSPResponseIsWeakHash(SecOCSPResponseRef response);

__END_DECLS

#endif /* !_SECURITY_SECOCSPRESPONSE_H_ */
