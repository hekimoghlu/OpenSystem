/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
#include "csrTemplates.h"
#include <stddef.h>
#include "SecAsn1Templates.h"
#include "keyTemplates.h"

const SecAsn1Template kSecAsn1CertRequestInfoTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSSCertRequestInfo)},
    {SEC_ASN1_INTEGER, offsetof(NSSCertRequestInfo, version)},
    {SEC_ASN1_INLINE, offsetof(NSSCertRequestInfo, subject), kSecAsn1NameTemplate},
    {SEC_ASN1_INLINE, offsetof(NSSCertRequestInfo, subjectPublicKeyInfo), kSecAsn1SubjectPublicKeyInfoTemplate},
    {SEC_ASN1_CONSTRUCTED | SEC_ASN1_CONTEXT_SPECIFIC | 0,
     offsetof(NSSCertRequestInfo, attributes),
     kSecAsn1SetOfAttributeTemplate},
    {0}};

const SecAsn1Template kSecAsn1CertRequestTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSSCertRequest)},
    {SEC_ASN1_INLINE, offsetof(NSSCertRequest, reqInfo), kSecAsn1CertRequestInfoTemplate},
    {SEC_ASN1_INLINE, offsetof(NSSCertRequest, signatureAlgorithm), kSecAsn1AlgorithmIDTemplate},
    {SEC_ASN1_BIT_STRING, offsetof(NSSCertRequest, signature)},
    {0}};

const SecAsn1Template kSecAsn1SignedCertRequestTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_SignedCertRequest)},
    {SEC_ASN1_ANY, offsetof(NSS_SignedCertRequest, certRequestBlob), kSecAsn1CertRequestInfoTemplate},
    {SEC_ASN1_INLINE, offsetof(NSS_SignedCertRequest, signatureAlgorithm), kSecAsn1AlgorithmIDTemplate},
    {SEC_ASN1_BIT_STRING, offsetof(NSS_SignedCertRequest, signature)},
    {0}};
