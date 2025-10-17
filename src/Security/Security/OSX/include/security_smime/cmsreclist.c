/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#include "cmslocal.h"

#include "cert.h"
#include "secitem.h"
#include "secoid.h"

#include <Security/SecIdentity.h>
#include <security_asn1/secasn1.h>
#include <security_asn1/secerr.h>
#include <utilities/SecCFWrappers.h>

static int nss_cms_recipients_traverse(SecCmsRecipientInfoRef* recipientinfos,
                                       SecCmsRecipient** recipient_list)
{
    int count = 0;
    int rlindex = 0;
    int i, j;
    SecCmsRecipient* rle;
    SecCmsRecipientInfoRef ri;
    SecCmsRecipientEncryptedKey* rek;

    for (i = 0; recipientinfos[i] != NULL; i++) {
        ri = recipientinfos[i];
        switch (ri->recipientInfoType) {
            case SecCmsRecipientInfoIDKeyTrans:
                if (recipient_list) {
                    /* alloc one & fill it out */
                    rle = (SecCmsRecipient*)PORT_ZAlloc(sizeof(SecCmsRecipient));
                    if (rle == NULL)
                        return -1;

                    rle->riIndex = i;
                    rle->subIndex = -1;
                    switch (ri->ri.keyTransRecipientInfo.recipientIdentifier.identifierType) {
                        case SecCmsRecipientIDIssuerSN:
                            rle->kind = RLIssuerSN;
                            rle->id.issuerAndSN =
                                ri->ri.keyTransRecipientInfo.recipientIdentifier.id.issuerAndSN;
                            break;
                        case SecCmsRecipientIDSubjectKeyID:
                            rle->kind = RLSubjKeyID;
                            rle->id.subjectKeyID =
                                ri->ri.keyTransRecipientInfo.recipientIdentifier.id.subjectKeyID;
                            break;
                    }
                    recipient_list[rlindex++] = rle;
                } else {
                    count++;
                }
                break;
            case SecCmsRecipientInfoIDKeyAgree:
                if (ri->ri.keyAgreeRecipientInfo.recipientEncryptedKeys == NULL)
                    break;
                for (j = 0; ri->ri.keyAgreeRecipientInfo.recipientEncryptedKeys[j] != NULL; j++) {
                    if (recipient_list) {
                        rek = ri->ri.keyAgreeRecipientInfo.recipientEncryptedKeys[j];
                        /* alloc one & fill it out */
                        rle = (SecCmsRecipient*)PORT_ZAlloc(sizeof(SecCmsRecipient));
                        if (rle == NULL)
                            return -1;

                        rle->riIndex = i;
                        rle->subIndex = j;
                        switch (rek->recipientIdentifier.identifierType) {
                            case SecCmsKeyAgreeRecipientIDIssuerSN:
                                rle->kind = RLIssuerSN;
                                rle->id.issuerAndSN = rek->recipientIdentifier.id.issuerAndSN;
                                break;
                            case SecCmsKeyAgreeRecipientIDRKeyID:
                                rle->kind = RLSubjKeyID;
                                rle->id.subjectKeyID =
                                    &rek->recipientIdentifier.id.recipientKeyIdentifier.subjectKeyIdentifier;
                                break;
                        }
                        recipient_list[rlindex++] = rle;
                    } else {
                        count++;
                    }
                }
                break;
            case SecCmsRecipientInfoIDKEK:
                /* KEK is not implemented */
                break;
        }
    }
    /* if we have a recipient list, we return on success (-1, above, on failure) */
    /* otherwise, we return the count. */
    if (recipient_list) {
        recipient_list[rlindex] = NULL;
        return 0;
    } else {
        return count;
    }
}

SecCmsRecipient** nss_cms_recipient_list_create(SecCmsRecipientInfoRef* recipientinfos)
{
    int count, rv;
    SecCmsRecipient** recipient_list;

    /* count the number of recipient identifiers */
    count = nss_cms_recipients_traverse(recipientinfos, NULL);
    if (count <= 0 || count >= (int)((INT_MAX / sizeof(SecCmsRecipient*)) - 1)) {
        /* no recipients? or risk of underallocation 20130783 */
        PORT_SetError(SEC_ERROR_BAD_DATA);
#if 0
	PORT_SetErrorString("Cannot find recipient data in envelope.");
#endif
        return NULL;
    }

    /* allocate an array of pointers */
    recipient_list = (SecCmsRecipient**)PORT_ZAlloc((size_t)(count + 1) * sizeof(SecCmsRecipient*));
    if (recipient_list == NULL) {
        return NULL;
    }

    /* now fill in the recipient_list */
    rv = nss_cms_recipients_traverse(recipientinfos, recipient_list);
    if (rv < 0) {
        nss_cms_recipient_list_destroy(recipient_list);
        return NULL;
    }
    return recipient_list;
}

void nss_cms_recipient_list_destroy(SecCmsRecipient** recipient_list)
{
    int i;
    SecCmsRecipient* recipient;

    for (i = 0; recipient_list[i] != NULL; i++) {
        recipient = recipient_list[i];
        CFReleaseNull(recipient->cert);
        CFReleaseNull(recipient->privkey);
#if 0
	// @@@ Eliminate slot stuff.
	if (recipient->slot)
	    PK11_FreeSlot(recipient->slot);
#endif
        PORT_Free(recipient);
        recipient_list[i] = NULL;
    }
    PORT_Free(recipient_list);
}

SecCmsRecipientEncryptedKey* SecCmsRecipientEncryptedKeyCreate(PLArenaPool* poolp)
{
    return (SecCmsRecipientEncryptedKey*)PORT_ArenaZAlloc(poolp, sizeof(SecCmsRecipientEncryptedKey));
}


int nss_cms_FindCertAndKeyByRecipientList(SecCmsRecipient** recipient_list, void* wincx)
{
    SecCmsRecipient* recipient = NULL;
    SecCertificateRef cert = NULL;
    SecPrivateKeyRef privKey = NULL;
    SecIdentityRef identity = NULL;
    CFTypeRef keychainOrArray = NULL;  // @@@ The caller should be able to pass this in somehow.
    int index;

    for (index = 0; recipient_list[index] != NULL; ++index) {
        recipient = recipient_list[index];

        switch (recipient->kind) {
            case RLIssuerSN:
                identity = CERT_FindIdentityByIssuerAndSN(keychainOrArray, recipient->id.issuerAndSN);
                break;
            case RLSubjKeyID:
                identity = CERT_FindIdentityBySubjectKeyID(keychainOrArray, recipient->id.subjectKeyID);
                break;
        }

        if (identity) {
            break;
        }
    }

    if (!recipient) {
        goto loser;
    }

    if (!identity || SecIdentityCopyCertificate(identity, &cert)) {
        goto loser;
    }
    if (!identity || SecIdentityCopyPrivateKey(identity, &privKey)) {
        goto loser;
    }
    CFReleaseNull(identity);

    recipient->cert = cert;
    recipient->privkey = privKey;

    return index;

loser:
    CFReleaseNull(identity);
    CFReleaseNull(cert);
    CFReleaseNull(privKey);

    return -1;
}
