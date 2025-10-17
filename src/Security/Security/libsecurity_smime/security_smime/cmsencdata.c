/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
 * CMS encryptedData methods.
 */

#include <Security/SecCmsEncryptedData.h>

#include <Security/SecCmsContentInfo.h>

#include "cmslocal.h"

#include "SecAsn1Item.h"
#include "secoid.h"

#include <security_asn1/secasn1.h>
#include <security_asn1/secerr.h>
#include <security_asn1/secport.h>
#include <utilities/SecCFWrappers.h>

/*
 * SecCmsEncryptedDataCreate - create an empty encryptedData object.
 *
 * "algorithm" specifies the bulk encryption algorithm to use.
 * "keysize" is the key size.
 * 
 * An error results in a return value of NULL and an error set.
 * (Retrieve specific errors via PORT_GetError()/XP_GetError().)
 */
SecCmsEncryptedDataRef SecCmsEncryptedDataCreate(SecCmsMessageRef cmsg, SECOidTag algorithm, int keysize)
{
    void* mark;
    SecCmsEncryptedDataRef encd;
    PLArenaPool* poolp;
#if 0
    SECAlgorithmID *pbe_algid;
#endif
    OSStatus rv;

    poolp = cmsg->poolp;

    mark = PORT_ArenaMark(poolp);

    encd = (SecCmsEncryptedDataRef)PORT_ArenaZAlloc(poolp, sizeof(SecCmsEncryptedData));
    if (encd == NULL)
        goto loser;

    encd->contentInfo.cmsg = cmsg;

    /* version is set in SecCmsEncryptedDataEncodeBeforeStart() */

    switch (algorithm) {
        /* XXX hmmm... hardcoded algorithms? */
        case SEC_OID_AES_128_CBC:
        case SEC_OID_AES_192_CBC:
        case SEC_OID_AES_256_CBC:
        case SEC_OID_RC2_CBC:
        case SEC_OID_DES_EDE3_CBC:
        case SEC_OID_DES_CBC:
            rv = SecCmsContentInfoSetContentEncAlg(&(encd->contentInfo), algorithm, NULL, keysize);
            break;
        default:
            /* Assume password-based-encryption.  At least, try that. */
#if 1
            // @@@ Fix me
            rv = SECFailure;
            break;
#else
            pbe_algid = PK11_CreatePBEAlgorithmID(algorithm, 1, NULL);
            if (pbe_algid == NULL) {
                rv = SECFailure;
                break;
            }
            rv = SecCmsContentInfoSetContentEncAlgID(&(encd->contentInfo), pbe_algid, keysize);
            SECOID_DestroyAlgorithmID(pbe_algid, PR_TRUE);
            break;
#endif
    }
    if (rv != SECSuccess)
        goto loser;

    PORT_ArenaUnmark(poolp, mark);
    return encd;

loser:
    PORT_ArenaRelease(poolp, mark);
    return NULL;
}

/*
 * SecCmsEncryptedDataDestroy - destroy an encryptedData object
 */
void SecCmsEncryptedDataDestroy(SecCmsEncryptedDataRef encd)
{
    if (encd == NULL) {
        return;
    }
    /* everything's in a pool, so don't worry about the storage */
    SecCmsContentInfoDestroy(&(encd->contentInfo));
    return;
}

/*
 * SecCmsEncryptedDataGetContentInfo - return pointer to encryptedData object's contentInfo
 */
SecCmsContentInfoRef SecCmsEncryptedDataGetContentInfo(SecCmsEncryptedDataRef encd)
{
    return &(encd->contentInfo);
}

/*
 * SecCmsEncryptedDataEncodeBeforeStart - do all the necessary things to a EncryptedData
 *     before encoding begins.
 *
 * In particular:
 *  - set the correct version value.
 *  - get the encryption key
 */
OSStatus SecCmsEncryptedDataEncodeBeforeStart(SecCmsEncryptedDataRef encd)
{
    int version;
    SecSymmetricKeyRef bulkkey = NULL;
    SecAsn1Item* dummy;
    SecCmsContentInfoRef cinfo = &(encd->contentInfo);

    if (SecCmsArrayIsEmpty((void**)encd->unprotectedAttr)) {
        version = SEC_CMS_ENCRYPTED_DATA_VERSION;
    } else {
        version = SEC_CMS_ENCRYPTED_DATA_VERSION_UPATTR;
    }

    dummy = SEC_ASN1EncodeInteger(encd->contentInfo.cmsg->poolp, &(encd->version), version);
    if (dummy == NULL) {
        return SECFailure;
    }

    /* now get content encryption key (bulk key) by using our cmsg callback */
    if (encd->contentInfo.cmsg->decrypt_key_cb) {
        bulkkey = (*encd->contentInfo.cmsg->decrypt_key_cb)(encd->contentInfo.cmsg->decrypt_key_cb_arg,
                                                            SecCmsContentInfoGetContentEncAlg(cinfo));
    }
    if (bulkkey == NULL) {
        return SECFailure;
    }

    /* store the bulk key in the contentInfo so that the encoder can find it */
    SecCmsContentInfoSetBulkKey(cinfo, bulkkey);
    CFReleaseNull(bulkkey); /* This assumes the decrypt_key_cb hands us a copy of the key --mb */

    return SECSuccess;
}

/*
 * SecCmsEncryptedDataEncodeBeforeData - set up encryption
 */
OSStatus SecCmsEncryptedDataEncodeBeforeData(SecCmsEncryptedDataRef encd)
{
    SecCmsContentInfoRef cinfo;
    SecSymmetricKeyRef bulkkey;
    SECAlgorithmID* algid;

    cinfo = &(encd->contentInfo);

    /* find bulkkey and algorithm - must have been set by SecCmsEncryptedDataEncodeBeforeStart */
    bulkkey = SecCmsContentInfoGetBulkKey(cinfo);
    if (bulkkey == NULL) {
        return SECFailure;
    }
    algid = SecCmsContentInfoGetContentEncAlg(cinfo);
    if (algid == NULL) {
        CFReleaseNull(bulkkey);
        return SECFailure;
    }

    /* this may modify algid (with IVs generated in a token).
     * it is therefore essential that algid is a pointer to the "real" contentEncAlg,
     * not just to a copy */
    cinfo->ciphcx = SecCmsCipherContextStartEncrypt(encd->contentInfo.cmsg->poolp, bulkkey, algid);
    CFReleaseNull(bulkkey);
    if (cinfo->ciphcx == NULL) {
        return SECFailure;
    }

    return SECSuccess;
}

/*
 * SecCmsEncryptedDataEncodeAfterData - finalize this encryptedData for encoding
 */
OSStatus SecCmsEncryptedDataEncodeAfterData(SecCmsEncryptedDataRef encd)
{
    if (encd->contentInfo.ciphcx) {
        SecCmsCipherContextDestroy(encd->contentInfo.ciphcx);
        encd->contentInfo.ciphcx = NULL;
    }

    /* nothing to do after data */
    return SECSuccess;
}


/*
 * SecCmsEncryptedDataDecodeBeforeData - find bulk key & set up decryption
 */
OSStatus SecCmsEncryptedDataDecodeBeforeData(SecCmsEncryptedDataRef encd)
{
    SecSymmetricKeyRef bulkkey = NULL;
    SecCmsContentInfoRef cinfo;
    SECAlgorithmID* bulkalg;
    OSStatus rv = SECFailure;

    cinfo = &(encd->contentInfo);

    bulkalg = SecCmsContentInfoGetContentEncAlg(cinfo);

    if (encd->contentInfo.cmsg->decrypt_key_cb == NULL) { /* no callback? no key../ */
        goto loser;
    }

    bulkkey = (*encd->contentInfo.cmsg->decrypt_key_cb)(encd->contentInfo.cmsg->decrypt_key_cb_arg, bulkalg);
    if (bulkkey == NULL) {
        /* no success finding a bulk key */
        goto loser;
    }

    SecCmsContentInfoSetBulkKey(cinfo, bulkkey);

    cinfo->ciphcx = SecCmsCipherContextStartDecrypt(bulkkey, bulkalg);
    if (cinfo->ciphcx == NULL) {
        goto loser; /* error has been set by SecCmsCipherContextStartDecrypt */
    }

#if 1
        // @@@ Not done yet
#else
    /* 
     * HACK ALERT!!
     * For PKCS5 Encryption Algorithms, the bulkkey is actually a different
     * structure.  Therefore, we need to set the bulkkey to the actual key 
     * prior to freeing it.
     */
    if (SEC_PKCS5IsAlgorithmPBEAlg(bulkalg)) {
        SEC_PKCS5KeyAndPassword* keyPwd = (SEC_PKCS5KeyAndPassword*)bulkkey;
        bulkkey = keyPwd->key;
    }
#endif
    rv = SECSuccess;

loser:
    CFReleaseNull(bulkkey);
    return rv;
}

/*
 * SecCmsEncryptedDataDecodeAfterData - finish decrypting this encryptedData's content
 */
OSStatus SecCmsEncryptedDataDecodeAfterData(SecCmsEncryptedDataRef encd)
{
    if (encd->contentInfo.ciphcx) {
        SecCmsCipherContextDestroy(encd->contentInfo.ciphcx);
        encd->contentInfo.ciphcx = NULL;
    }

    return SECSuccess;
}

/*
 * SecCmsEncryptedDataDecodeAfterEnd - finish decoding this encryptedData
 */
OSStatus SecCmsEncryptedDataDecodeAfterEnd(SecCmsEncryptedDataRef encd)
{
    /* apply final touches */
    return SECSuccess;
}
