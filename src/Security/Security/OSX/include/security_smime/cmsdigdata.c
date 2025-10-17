/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
 * CMS digestedData methods.
 */

#include <Security/SecCmsDigestedData.h>

#include <Security/SecCmsContentInfo.h>
#include <Security/SecCmsDigestContext.h>

#include "cmslocal.h"

#include <security_asn1/secasn1.h>
#include <security_asn1/secerr.h>
#include "secitem.h"
#include "secoid.h"

/*
 * SecCmsDigestedDataCreate - create a digestedData object (presumably for encoding)
 *
 * version will be set by SecCmsDigestedDataEncodeBeforeStart
 * digestAlg is passed as parameter
 * contentInfo must be filled by the user
 * digest will be calculated while encoding
 */
SecCmsDigestedDataRef SecCmsDigestedDataCreate(SecCmsMessageRef cmsg, SECAlgorithmID* digestalg)
{
    void* mark;
    SecCmsDigestedDataRef digd;
    PLArenaPool* poolp;

    poolp = cmsg->poolp;

    mark = PORT_ArenaMark(poolp);

    digd = (SecCmsDigestedDataRef)PORT_ArenaZAlloc(poolp, sizeof(SecCmsDigestedData));
    if (digd == NULL)
        goto loser;

    digd->cmsg = cmsg;

    if (SECOID_CopyAlgorithmID(poolp, &(digd->digestAlg), digestalg) != SECSuccess)
        goto loser;

    PORT_ArenaUnmark(poolp, mark);
    return digd;

loser:
    PORT_ArenaRelease(poolp, mark);
    return NULL;
}

/*
 * SecCmsDigestedDataDestroy - destroy a digestedData object
 */
void SecCmsDigestedDataDestroy(SecCmsDigestedDataRef digd)
{
    if (digd == NULL) {
        return;
    }
    /* everything's in a pool, so don't worry about the storage */
    SecCmsContentInfoDestroy(&(digd->contentInfo));
    return;
}

/*
 * SecCmsDigestedDataGetContentInfo - return pointer to digestedData object's contentInfo
 */
SecCmsContentInfoRef SecCmsDigestedDataGetContentInfo(SecCmsDigestedDataRef digd)
{
    return &(digd->contentInfo);
}

/*
 * SecCmsDigestedDataEncodeBeforeStart - do all the necessary things to a DigestedData
 *     before encoding begins.
 *
 * In particular:
 *  - set the right version number. The contentInfo's content type must be set up already.
 */
OSStatus SecCmsDigestedDataEncodeBeforeStart(SecCmsDigestedDataRef digd)
{
    long version;
    CSSM_DATA_PTR dummy;

    version = SEC_CMS_DIGESTED_DATA_VERSION_DATA;
    if (SecCmsContentInfoGetContentTypeTag(&(digd->contentInfo)) != SEC_OID_PKCS7_DATA)
        version = SEC_CMS_DIGESTED_DATA_VERSION_ENCAP;

    dummy = SEC_ASN1EncodeInteger(digd->cmsg->poolp, &(digd->version), version);
    return (dummy == NULL) ? SECFailure : SECSuccess;
}

/*
 * SecCmsDigestedDataEncodeBeforeData - do all the necessary things to a DigestedData
 *     before the encapsulated data is passed through the encoder.
 *
 * In detail:
 *  - set up the digests if necessary
 */
OSStatus SecCmsDigestedDataEncodeBeforeData(SecCmsDigestedDataRef digd)
{
    /* set up the digests */
    if (digd->digestAlg.algorithm.Length != 0 && digd->digest.Length == 0) {
        /* if digest is already there, do nothing */
        digd->contentInfo.digcx = SecCmsDigestContextStartSingle(&(digd->digestAlg));
        if (digd->contentInfo.digcx == NULL)
            return SECFailure;
    }
    return SECSuccess;
}

/*
 * SecCmsDigestedDataEncodeAfterData - do all the necessary things to a DigestedData
 *     after all the encapsulated data was passed through the encoder.
 *
 * In detail:
 *  - finish the digests
 */
OSStatus SecCmsDigestedDataEncodeAfterData(SecCmsDigestedDataRef digd)
{
    OSStatus rv = SECSuccess;
    /* did we have digest calculation going on? */
    if (digd->contentInfo.digcx) {
        rv = SecCmsDigestContextFinishSingle(
            digd->contentInfo.digcx, (SecArenaPoolRef)digd->cmsg->poolp, &(digd->digest));
        /* error has been set by SecCmsDigestContextFinishSingle */
        digd->contentInfo.digcx = NULL;
    }

    return rv;
}

/*
 * SecCmsDigestedDataDecodeBeforeData - do all the necessary things to a DigestedData
 *     before the encapsulated data is passed through the encoder.
 *
 * In detail:
 *  - set up the digests if necessary
 */
OSStatus SecCmsDigestedDataDecodeBeforeData(SecCmsDigestedDataRef digd)
{
    /* is there a digest algorithm yet? */
    if (digd->digestAlg.algorithm.Length == 0)
        return SECFailure;

    digd->contentInfo.digcx = SecCmsDigestContextStartSingle(&(digd->digestAlg));
    if (digd->contentInfo.digcx == NULL)
        return SECFailure;

    return SECSuccess;
}

/*
 * SecCmsDigestedDataDecodeAfterData - do all the necessary things to a DigestedData
 *     after all the encapsulated data was passed through the encoder.
 *
 * In detail:
 *  - finish the digests
 */
OSStatus SecCmsDigestedDataDecodeAfterData(SecCmsDigestedDataRef digd)
{
    OSStatus rv = SECSuccess;
    /* did we have digest calculation going on? */
    if (digd->contentInfo.digcx) {
        rv = SecCmsDigestContextFinishSingle(
            digd->contentInfo.digcx, (SecArenaPoolRef)digd->cmsg->poolp, &(digd->cdigest));
        /* error has been set by SecCmsDigestContextFinishSingle */
        digd->contentInfo.digcx = NULL;
    }

    return rv;
}

/*
 * SecCmsDigestedDataDecodeAfterEnd - finalize a digestedData.
 *
 * In detail:
 *  - check the digests for equality
 */
OSStatus SecCmsDigestedDataDecodeAfterEnd(SecCmsDigestedDataRef digd)
{
    if (!digd) {
        return SECFailure;
    }
    /* did we have digest calculation going on? */
    if (digd->cdigest.Length != 0) {
        /* XXX comparision btw digest & cdigest */
        /* XXX set status */
        /* TODO!!!! */
    }

    return SECSuccess;
}
