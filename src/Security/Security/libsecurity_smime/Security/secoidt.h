/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#ifndef _SECOIDT_H_
#define _SECOIDT_H_
/*
 * secoidt.h - public data structures for ASN.1 OID functions
 */

#include <Security/SecCmsBase.h>

typedef enum {
    INVALID_CERT_EXTENSION = 0,
    UNSUPPORTED_CERT_EXTENSION = 1,
    SUPPORTED_CERT_EXTENSION = 2
} SECSupportExtenTag;

struct SECOidDataStr {
    SecAsn1Item oid;
    SECOidTag offset;
    const char* desc;
#if USE_CDSA_CRYPTO
    SecAsn1AlgId cssmAlgorithm;
#endif
    SECSupportExtenTag supportedExtension;
    /* only used for x.509 v3 extensions, so
				   that we can print the names of those
				   extensions that we don't even support */
};

#endif /* _SECOIDT_H_ */
