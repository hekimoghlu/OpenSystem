/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#include <libDER/asn1Types.h>
#include <libDER/DER_Decode.h>
#include <AssertMacros.h>
#include <Security/cssmtype.h>
#include <stdlib.h>
#include "tsaDERUtilities.h"

#ifndef DER_MULTIBYTE_TAGS
#error We expect DER_MULTIBYTE_TAGS
#endif

/* PKIStatusInfo */
typedef struct {
    DERItem     status;         // INTEGER
	DERItem     statusString;      // UTF8_STRING | SEC_ASN1_OPTIONAL
    DERItem     failInfo;          // BIT_STRING | SEC_ASN1_OPTIONAL
} DERPKIStatusInfo;

/* xx */
typedef struct {
	DERItem     statusString;      // UTF8_STRING | SEC_ASN1_OPTIONAL
} DERPKIStatusStringInner;

/* TimeStampResp */
typedef struct
{
    DERItem status;             /* PKIStatusInfo */
    DERItem timeStampToken;     /* TimeStampToken */
} DERTimeStampResp;

/* TimeStampResp */
const DERItemSpec DERTimeStampRespItemSpecs[] = 
{
    { DER_OFFSET(DERTimeStampResp, status),
        ASN1_CONSTR_SEQUENCE, DER_DEC_NO_OPTS },
    { DER_OFFSET(DERTimeStampResp, timeStampToken),
        ASN1_CONSTR_SEQUENCE, DER_DEC_NO_OPTS | DER_DEC_OPTIONAL | DER_DEC_SAVE_DER}
};
const DERSize DERNumTimeStampRespItemSpecs = sizeof(DERTimeStampRespItemSpecs) / sizeof(DERItemSpec);

/*
    This code is here rather than in libsecurity_smime because
    libsecurity_smime doesn't know about libDER
*/

int DERDecodeTimeStampResponse(
	const CSSM_DATA *contents,
    CSSM_DATA *derStatus,
    CSSM_DATA *derTimeStampToken,
	size_t			*numUsedBytes)      /* RETURNED */
{
    DERReturn drtn = DR_ParamErr;
    DERDecodedInfo decodedPackage = {};

    if (contents)
    {
        DERItem derContents = {.data = contents->Data, .length = contents->Length };
        DERTimeStampResp derResponse = {{0,},{0,}};
        DERReturn rx;
        require_noerr(DERDecodeItem(&derContents, &decodedPackage), badResponse);

        rx = DERParseSequenceContent(&decodedPackage.content,
            DERNumTimeStampRespItemSpecs, DERTimeStampRespItemSpecs, 
            &derResponse, 0);
        if (rx != DR_Success)
            goto badResponse;
/*
        require_noerr(DERParseSequenceContent(&decodedPackage.content,
            DERNumTimeStampRespItemSpecs, DERTimeStampRespItemSpecs, 
            &derResponse, 0), badResponse);
*/
        if (derStatus && derResponse.status.data)
        {
            derStatus->Data = malloc(derResponse.status.length);
            derStatus->Length = derResponse.status.length;
            memcpy(derStatus->Data, derResponse.status.data, derStatus->Length);
        }
        if (derTimeStampToken && derResponse.timeStampToken.data)
        {
            derTimeStampToken->Data = malloc(derResponse.timeStampToken.length);
            derTimeStampToken->Length = derResponse.timeStampToken.length;
            memcpy(derTimeStampToken->Data, derResponse.timeStampToken.data, derTimeStampToken->Length);
        }
    }

    drtn = DR_Success;
    
badResponse:
    if (numUsedBytes)
        *numUsedBytes = decodedPackage.content.length +
            decodedPackage.content.data - (contents ? contents->Data : 0);

    return drtn;
}

