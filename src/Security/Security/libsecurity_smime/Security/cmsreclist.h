/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#ifndef _CMSRECLIST_H
#define _CMSRECLIST_H

struct SecCmsRecipientStr {
    int riIndex;  /* this recipient's index in recipientInfo array */
    int subIndex; /* index into recipientEncryptedKeys */
                  /* (only in SecCmsKeyAgreeRecipientInfoStr) */
    enum {
        RLIssuerSN = 0,
        RLSubjKeyID = 1
    } kind; /* for conversion recipientinfos -> recipientlist */
    union {
        SecCmsIssuerAndSN* issuerAndSN;
        SecAsn1Item* subjectKeyID;
    } id;

    /* result data (filled out for each recipient that's us) */
    SecCertificateRef cert;
    SecPrivateKeyRef privkey;
    //PK11SlotInfo *		slot;
};

typedef struct SecCmsRecipientStr SecCmsRecipient;

int nss_cms_FindCertAndKeyByRecipientList(SecCmsRecipient** recipient_list, void* wincx);

#endif /* _CMSRECLIST_H */
