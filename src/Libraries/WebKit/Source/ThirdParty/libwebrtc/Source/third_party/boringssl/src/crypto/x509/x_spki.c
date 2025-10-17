/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
// This module was send to me my Pat Richards <patr@x509.com> who wrote it.
// It is under my Copyright with his permission.

#include <openssl/asn1t.h>
#include <openssl/x509.h>

#include "internal.h"


ASN1_SEQUENCE(NETSCAPE_SPKAC) = {
    ASN1_SIMPLE(NETSCAPE_SPKAC, pubkey, X509_PUBKEY),
    ASN1_SIMPLE(NETSCAPE_SPKAC, challenge, ASN1_IA5STRING),
} ASN1_SEQUENCE_END(NETSCAPE_SPKAC)

IMPLEMENT_ASN1_FUNCTIONS_const(NETSCAPE_SPKAC)

ASN1_SEQUENCE(NETSCAPE_SPKI) = {
    ASN1_SIMPLE(NETSCAPE_SPKI, spkac, NETSCAPE_SPKAC),
    ASN1_SIMPLE(NETSCAPE_SPKI, sig_algor, X509_ALGOR),
    ASN1_SIMPLE(NETSCAPE_SPKI, signature, ASN1_BIT_STRING),
} ASN1_SEQUENCE_END(NETSCAPE_SPKI)

IMPLEMENT_ASN1_FUNCTIONS_const(NETSCAPE_SPKI)
