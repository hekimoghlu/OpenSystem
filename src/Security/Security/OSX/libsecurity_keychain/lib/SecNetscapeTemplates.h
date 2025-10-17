/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#ifndef	_SEC_IMPORT_EXPORT_NETSCAPE_TEMPLATES_H_
#define _SEC_IMPORT_EXPORT_NETSCAPE_TEMPLATES_H_

#include <Security/secasn1t.h>
#include <Security/cssmtype.h>
#include <Security/X509Templates.h>
#include <Security/keyTemplates.h>
#include <Security/x509defs.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Netscape Certifiate Sequence is defined by Netscape as a PKCS7
 * ContentInfo with a contentType of netscape-cert-sequence and a content
 * consisting of a sequence of certificates.
 *
 * For simplicity - i.e., to avoid the general purpose ContentInfo
 * polymorphism - we'll just hard-code this particular type right here.
 *
 * Inside the ContentInfo is an array of standard X509 certificates.
 * We don't need to parse the certs themselves so they remain as 
 * opaque data blobs. 
 */
typedef struct {
	CSSM_OID		contentType;		// netscape-cert-sequence
	CSSM_DATA		**certs;
} NetscapeCertSequence;

extern const SecAsn1Template NetscapeCertSequenceTemplate[];

/*
 * Public key/challenge, to send to CA.
 *
 * PublicKeyAndChallenge ::= SEQUENCE {
 *
 * Â  	spki SubjectPublicKeyInfo,
 *   	challenge IA5STRING
 * }
 *
 * SignedPublicKeyAndChallenge ::= SEQUENCE {
 * 		publicKeyAndChallenge PublicKeyAndChallenge,
 *		signatureAlgorithm AlgorithmIdentifier,
 *		signature BIT STRING
 * }
 */
typedef struct {
	CSSM_X509_SUBJECT_PUBLIC_KEY_INFO	spki;
	CSSM_DATA							challenge;	// ASCII
} PublicKeyAndChallenge;

typedef struct {
	PublicKeyAndChallenge				pubKeyAndChallenge;
	CSSM_X509_ALGORITHM_IDENTIFIER		algId;
	CSSM_DATA							signature; // length in BITS
} SignedPublicKeyAndChallenge;

extern const SecAsn1Template PublicKeyAndChallengeTemplate[];
extern const SecAsn1Template SignedPublicKeyAndChallengeTemplate[];

#ifdef __cplusplus
}
#endif

#endif	/* _SEC_IMPORT_EXPORT_NETSCAPE_TEMPLATES_H_ */

