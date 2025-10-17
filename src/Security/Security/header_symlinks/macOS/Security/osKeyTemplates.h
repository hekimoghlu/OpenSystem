/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
 * osKeyTemplate.h -  ASN1 templates for openssl asymmetric keys
 */

#ifndef	_OS_KEY_TEMPLATES_H_
#define _OS_KEY_TEMPLATES_H_

#include <Security/keyTemplates.h>

/*
 * Arrays of SecAsn1Template are always associated with a specific
 * C struct. We attempt to use C structs which are defined in CDSA
 * if at all possible; these always start with the CSSM_ prefix.
 * Otherwise we define the struct here, with an NSS_ prefix.
 * In either case, the name of the C struct is listed in comments
 * along with the extern declaration of the SecAsn1Template array.
 */

#ifdef	__cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/*** 
 *** Note: RSA and Diffie-Hellman keys and structs are in 
 *** security_asn1/keyTemplates.h.
 ***/
 
// MARK: *** DSA ***

/* 
 * Note that most of the DSA structs are hand rolled and are not
 * expressed in ASN1 in any doc that I'm aware of.
 */
 
/**** 
 **** DSA support 
 ****/

/* 
 * DSA algorithm parameters. Used in CDSA key generation context as 
 * well as the parameters in an X509-formatted DSA public key.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item	p;
	SecAsn1Item	q;
	SecAsn1Item	g;
} NSS_DSAAlgParams SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAAlgParamsTemplate[] SEC_ASN1_API_DEPRECATED;

/*
 * DSA algorithm parameters, BSAFE style. Only used in FIPS186 format
 * public and private keys.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item	keySizeInBits;
	SecAsn1Item	p;
	SecAsn1Item	q;
	SecAsn1Item	g;
} NSS_DSAAlgParamsBSAFE SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAAlgParamsBSAFETemplate[] SEC_ASN1_API_DEPRECATED;

/*
 * DSA X509-style AlgorithmID. Avoids ASN_ANY processing via direct 
 * insertion of the appropriate parameters.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Oid			algorithm;
	NSS_DSAAlgParams	*params;		// optional
} NSS_DSAAlgorithmIdX509 SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAAlgorithmIdX509Template[] SEC_ASN1_API_DEPRECATED;

/*
 * DSA AlgorithmID, BSAFE style. Avoids ASN_ANY 
 * processing via direct insertion of the appropriate parameters.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Oid				algorithm;
	NSS_DSAAlgParamsBSAFE	params;
} NSS_DSAAlgorithmIdBSAFE SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAAlgorithmIdBSAFETemplate[] SEC_ASN1_API_DEPRECATED;

/**** 
 **** DSA public keys 
 ****/

/*
 * DSA public key, openssl/X509 format.
 *
 * The publicKey is actually the DER encoding of an ASN 
 * integer, wrapped in a BIT STRING. 
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	NSS_DSAAlgorithmIdX509	dsaAlg;
	SecAsn1Item				publicKey;		// BIT string - Length in bits
} NSS_DSAPublicKeyX509 SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAPublicKeyX509Template[] SEC_ASN1_API_DEPRECATED;

/*
 * DSA public key, BSAFE/FIPS186 format.
 * The public key is the DER encoding of an ASN integer, wrapped
 * in a bit string.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	NSS_DSAAlgorithmIdBSAFE		dsaAlg;
	SecAsn1Item					publicKey;	// BIT string - Length in bits
} NSS_DSAPublicKeyBSAFE SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAPublicKeyBSAFETemplate[] SEC_ASN1_API_DEPRECATED;

/**** 
 **** DSA private keys 
 ****/
 
/*
 * DSA Private key, openssl custom format.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item	version;
	SecAsn1Item	p;
	SecAsn1Item	q;
	SecAsn1Item	g;
	SecAsn1Item	pub;
	SecAsn1Item	priv;
} NSS_DSAPrivateKeyOpenssl SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAPrivateKeyOpensslTemplate[] SEC_ASN1_API_DEPRECATED;

/*
 * DSA private key, BSAFE/FIPS186 style.
 * This is basically a DSA-specific NSS_PrivateKeyInfo.
 *
 * NSS_DSAPrivateKeyBSAFE.privateKey is an octet string containing
 * the DER encoding of this.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item				privateKey;
} NSS_DSAPrivateKeyOcts SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAPrivateKeyOctsTemplate[] SEC_ASN1_API_DEPRECATED;

typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item				version;
	NSS_DSAAlgorithmIdBSAFE	dsaAlg;
	/* octet string containing a DER-encoded NSS_DSAPrivateKeyOcts */
	SecAsn1Item				privateKey;
} NSS_DSAPrivateKeyBSAFE SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAPrivateKeyBSAFETemplate[] SEC_ASN1_API_DEPRECATED;

/*
 * DSA Private Key, PKCS8/SMIME style. Doesn't have keySizeInBits
 * in the alg params; has version in the top-level struct; the 
 * private key itself is a DER-encoded integer wrapped in an
 * octet string.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item				version;
	NSS_DSAAlgorithmIdX509	dsaAlg;
	/* octet string containing DER-encoded integer */
	SecAsn1Item				privateKey;
    NSS_Attribute 			**attributes;		// optional
} NSS_DSAPrivateKeyPKCS8 SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSAPrivateKeyPKCS8Template[] SEC_ASN1_API_DEPRECATED;

/* 
 * DSA Signature.
 */
typedef struct SEC_ASN1_API_DEPRECATED {
	SecAsn1Item	r;
	SecAsn1Item	s;
} NSS_DSASignature SEC_ASN1_API_DEPRECATED;

extern const SecAsn1Template kSecAsn1DSASignatureTemplate[] SEC_ASN1_API_DEPRECATED;

#pragma clang diagnostic pop

#ifdef	__cplusplus
}
#endif


#endif	/* _OS_KEY_TEMPLATES_H_ */
