/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#include "SecNetscapeTemplates.h"
#include <Security/SecAsn1Templates.h>
#include <Security/secasn1t.h>
#include <stddef.h>

const SecAsn1Template NetscapeCertSequenceTemplate[] = {
    { SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(NetscapeCertSequence) },
	{ SEC_ASN1_OBJECT_ID,
	  offsetof(NetscapeCertSequence, contentType), 0},
   { SEC_ASN1_EXPLICIT | SEC_ASN1_CONSTRUCTED | 
		SEC_ASN1_CONTEXT_SPECIFIC | 0 , 
	    offsetof(NetscapeCertSequence, certs),
	    kSecAsn1SequenceOfAnyTemplate },
	{ 0 }
};
const SecAsn1Template PublicKeyAndChallengeTemplate[] = {
    { SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(PublicKeyAndChallenge) },
    { SEC_ASN1_INLINE,
	  offsetof(PublicKeyAndChallenge, spki),
	  kSecAsn1SubjectPublicKeyInfoTemplate },
    { SEC_ASN1_INLINE,
	  offsetof(PublicKeyAndChallenge, challenge),
	  kSecAsn1IA5StringTemplate },
	{ 0 }
};

extern const SecAsn1Template SignedPublicKeyAndChallengeTemplate[] = {
    { SEC_ASN1_SEQUENCE,
      0, NULL, sizeof(SignedPublicKeyAndChallenge) },
    { SEC_ASN1_INLINE,
	  offsetof(SignedPublicKeyAndChallenge, pubKeyAndChallenge),
	  PublicKeyAndChallengeTemplate },
    { SEC_ASN1_INLINE,
	  offsetof(SignedPublicKeyAndChallenge, algId),
	  kSecAsn1AlgorithmIDTemplate },
    { SEC_ASN1_BIT_STRING,
	  offsetof(SignedPublicKeyAndChallenge, signature) },
	{ 0 }
};

