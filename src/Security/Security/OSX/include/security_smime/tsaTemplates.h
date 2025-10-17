/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#ifndef _TSA_TEMPLATES_H_
#define _TSA_TEMPLATES_H_

#include <Security/X509Templates.h> /* NSS_CertExtension */
#include <Security/nameTemplates.h> /* NSS_GeneralName and support */
#include <Security/secasn1t.h>
#include <Security/x509defs.h> /* CSSM_X509_ALGORITHM_IDENTIFIER */
#include "cmstpriv.h"          /* SecCmsContentInfo */

#ifdef __cplusplus
extern "C" {
#endif

#pragma mark----- TSA Request -----


typedef CSSM_OID TSAPolicyId;

typedef struct {
    CSSM_X509_ALGORITHM_IDENTIFIER hashAlgorithm;
    CSSM_DATA hashedMessage;
} SecAsn1TSAMessageImprint;

typedef struct {
    CSSM_DATA seconds;  // INTEGER optional
    CSSM_DATA millis;   // INTEGER optional
    CSSM_DATA micros;   // INTEGER optional
} SecAsn1TSAAccuracy;

typedef struct {
    CSSM_DATA version;  // INTEGER (1)
    SecAsn1TSAMessageImprint messageImprint;
    TSAPolicyId reqPolicy;              // OPTIONAL
    CSSM_DATA nonce;                    // INTEGER optional
    CSSM_DATA certReq;                  // BOOL
    CSSM_X509_EXTENSIONS** extensions;  // [0] IMPLICIT Extensions OPTIONAL
} SecAsn1TSATimeStampReq;

#pragma mark----- TSA Response -----

typedef struct {
    CSSM_DATA status;
    CSSM_DATA statusString;  // OPTIONAL
    CSSM_DATA failInfo;      // OPTIONAL
} SecAsn1TSAPKIStatusInfo;

typedef SecCmsContentInfo SecTimeStampToken;

typedef struct {
    SecAsn1TSAPKIStatusInfo status;
    SecTimeStampToken timeStampToken;  // OPTIONAL
} SecAsn1TimeStampResp;

/*
    We use this to grab the raw DER, but not decode it for subsequent
    re-insertion into a CMS message as an unsigned attribute
*/

typedef struct {
    SecAsn1TSAPKIStatusInfo status;
    CSSM_DATA timeStampTokenDER;  // OPTIONAL
} SecAsn1TimeStampRespDER;

typedef struct {
    CSSM_DATA version;      // DEFAULT 1    *****
    TSAPolicyId reqPolicy;  // OPTIONAL
    SecAsn1TSAMessageImprint messageImprint;
    CSSM_DATA serialNumber;  // INTEGER
    CSSM_DATA genTime;
    SecAsn1TSAAccuracy accuracy;        // OPTIONAL
    CSSM_DATA ordering;                 // BOOLEAN DEFAULT FALSE
    CSSM_DATA nonce;                    // INTEGER optional
    CSSM_DATA tsa;                      // [0] GeneralName         OPTIONAL
    CSSM_X509_EXTENSIONS** extensions;  // [1] IMPLICIT Extensions OPTIONAL
} SecAsn1TSATSTInfo;

typedef enum {
    PKIS_Granted = 0,
    PKIS_GrantedWithMods = 1,
    PKIS_Rejection = 2,
    PKIS_Waiting = 3,
    PKIS_RevocationWarning = 4,
    PKIS_RevocationNotification = 5
} SecAsn1TSAPKIStatus;

typedef enum {
    FI_BadAlg = 0,
    FI_BadRequest = 2,
    FI_BadDataFormat = 5,
    FI_TimeNotAvailable = 14,
    FI_UnacceptedPolicy = 15,
    FI_UnacceptedExtension = 16,
    FI_AddInfoNotAvailable = 17,
    FI_SystemFailure = 25
} SecAsn1TSAPKIFailureInfo;


#ifdef __cplusplus
}
#endif

#endif /* _TSA_TEMPLATES_H_ */
