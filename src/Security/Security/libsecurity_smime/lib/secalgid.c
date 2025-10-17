/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#include <security_asn1/secasn1.h>
#include <security_asn1/secerr.h>
#include <security_asn1/secport.h>
#include "SecAsn1Item.h"
#include "secoid.h"

const SecAsn1Template SECOID_AlgorithmIDTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(SECAlgorithmID)},
    {
        SEC_ASN1_OBJECT_ID,
        offsetof(SECAlgorithmID, algorithm),
    },
    {
        SEC_ASN1_OPTIONAL | SEC_ASN1_ANY,
        offsetof(SECAlgorithmID, parameters),
    },
    {0}};

SECOidTag SECOID_GetAlgorithmTag(const SECAlgorithmID* id)
{
    if (id == NULL || id->algorithm.Data == NULL)
        return SEC_OID_UNKNOWN;

    return SECOID_FindOIDTag(&(id->algorithm));
}

SECStatus SECOID_SetAlgorithmID(PRArenaPool* arena, SECAlgorithmID* id, SECOidTag which, const SecAsn1Item* params)
{
    SECOidData* oiddata;
    Boolean add_null_param;

    oiddata = SECOID_FindOIDByTag(which);
    if (!oiddata) {
        PORT_SetError(SEC_ERROR_INVALID_ALGORITHM);
        return SECFailure;
    }

    if (SECITEM_CopyItem(arena, &id->algorithm, &oiddata->oid))
        return SECFailure;

    switch (which) {
        case SEC_OID_MD2:
        case SEC_OID_MD4:
        case SEC_OID_MD5:
        case SEC_OID_SHA1:
        case SEC_OID_SHA256:
        case SEC_OID_SHA384:
        case SEC_OID_SHA512:
        case SEC_OID_PKCS1_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_MD2_WITH_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_MD4_WITH_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_MD5_WITH_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_SHA1_WITH_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_SHA256_WITH_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_SHA384_WITH_RSA_ENCRYPTION:
        case SEC_OID_PKCS1_SHA512_WITH_RSA_ENCRYPTION:
            add_null_param = PR_TRUE;
            break;
        default:
            add_null_param = PR_FALSE;
            break;
    }

    if (params) {
        /*
		* I am specifically *not* enforcing the following assertion
		* (by following it up with an error and a return of failure)
		* because I do not want to introduce any change in the current
		* behavior.  But I do want for us to notice if the following is
		* ever true, because I do not think it should be so and probably
		* signifies an error/bug somewhere.
		*/
        PORT_Assert(!add_null_param || (params->Length == 2 && params->Data[0] == SEC_ASN1_NULL &&
                                        params->Data[1] == 0));
        if (SECITEM_CopyItem(arena, &id->parameters, params)) {
            return SECFailure;
        }
    } else {
        /*
		* Again, this is not considered an error.  But if we assume
		* that nobody tries to set the parameters field themselves
		* (but always uses this routine to do that), then we should
		* not hit the following assertion.  Unless they forgot to zero
		* the structure, which could also be a bad (and wrong) thing.
		*/
        PORT_Assert(id->parameters.Data == NULL);

        if (add_null_param) {
            (void)SECITEM_AllocItem(arena, &id->parameters, 2);
            if (id->parameters.Data == NULL) {
                return SECFailure;
            }
            id->parameters.Data[0] = SEC_ASN1_NULL;
            id->parameters.Data[1] = 0;
        }
    }

    return SECSuccess;
}

SECStatus SECOID_CopyAlgorithmID(PRArenaPool* arena, SECAlgorithmID* to, const SECAlgorithmID* from)
{
    SECStatus rv;

    rv = SECITEM_CopyItem(arena, &to->algorithm, &from->algorithm);
    if (rv)
        return rv;
    rv = SECITEM_CopyItem(arena, &to->parameters, &from->parameters);
    return rv;
}

void SECOID_DestroyAlgorithmID(SECAlgorithmID* algid, Boolean freeit)
{
    SECITEM_FreeItem(&algid->parameters, PR_FALSE);
    SECITEM_FreeItem(&algid->algorithm, PR_FALSE);
    if (freeit == PR_TRUE)
        PORT_Free(algid);
}

SECComparison SECOID_CompareAlgorithmID(const SECAlgorithmID* a, const SECAlgorithmID* b)
{
    SECComparison rv;

    rv = SECITEM_CompareItem(&a->algorithm, &b->algorithm);
    if (rv)
        return rv;
    rv = SECITEM_CompareItem(&a->parameters, &b->parameters);
    return rv;
}

/* This functions simply returns the address of the above-declared template. */
SEC_ASN1_CHOOSER_IMPLEMENT(SECOID_AlgorithmIDTemplate)
