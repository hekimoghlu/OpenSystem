/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "osKeyTemplates.h"
#include <stddef.h>

/**** 
 **** DSA support 
 ****/

/* X509 style DSA algorithm parameters */
const SecAsn1Template kSecAsn1DSAAlgParamsTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAAlgParams)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParams, p)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParams, q)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParams, g)},
    {
        0,
    }};

/* BSAFE style DSA algorithm parameters */
const SecAsn1Template kSecAsn1DSAAlgParamsBSAFETemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAAlgParamsBSAFE)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParamsBSAFE, keySizeInBits)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParamsBSAFE, p)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParamsBSAFE, q)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAAlgParamsBSAFE, g)},
    {
        0,
    }};

/* DSA X509-style AlgorithmID */
const SecAsn1Template kSecAsn1DSAAlgorithmIdX509Template[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAAlgorithmIdX509)},
    {SEC_ASN1_OBJECT_ID, offsetof(NSS_DSAAlgorithmIdX509, algorithm)},
    /* per CMS, this is optional */
    {SEC_ASN1_POINTER | SEC_ASN1_OPTIONAL, offsetof(NSS_DSAAlgorithmIdX509, params), kSecAsn1DSAAlgParamsTemplate},
    {
        0,
    }};

/* DSA BSAFE-style AlgorithmID */
const SecAsn1Template kSecAsn1DSAAlgorithmIdBSAFETemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAAlgorithmIdBSAFE)},
    {SEC_ASN1_OBJECT_ID, offsetof(NSS_DSAAlgorithmIdBSAFE, algorithm)},
    {SEC_ASN1_INLINE, offsetof(NSS_DSAAlgorithmIdBSAFE, params), kSecAsn1DSAAlgParamsBSAFETemplate},
    {
        0,
    }};

/**** 
 **** DSA public keys 
 ****/

/* DSA public key, openssl/X509 format */
const SecAsn1Template kSecAsn1DSAPublicKeyX509Template[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAPublicKeyX509)},
    {SEC_ASN1_INLINE, offsetof(NSS_DSAPublicKeyX509, dsaAlg), kSecAsn1DSAAlgorithmIdX509Template},
    {
        SEC_ASN1_BIT_STRING,
        offsetof(NSS_DSAPublicKeyX509, publicKey),
    },
    {
        0,
    }};

/* DSA public key, BSAFE/FIPS186 format */
const SecAsn1Template kSecAsn1DSAPublicKeyBSAFETemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAPublicKeyBSAFE)},
    {SEC_ASN1_INLINE, offsetof(NSS_DSAPublicKeyBSAFE, dsaAlg), kSecAsn1DSAAlgorithmIdBSAFETemplate},
    {
        SEC_ASN1_BIT_STRING,
        offsetof(NSS_DSAPublicKeyBSAFE, publicKey),
    },
    {
        0,
    }};

/**** 
 **** DSA private keys 
 ****/

/* DSA Private key, openssl custom format */
const SecAsn1Template kSecAsn1DSAPrivateKeyOpensslTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAPrivateKeyOpenssl)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOpenssl, version)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOpenssl, p)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOpenssl, q)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOpenssl, g)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOpenssl, pub)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOpenssl, priv)},
    {
        0,
    }};

/*
 * DSA private key, BSAFE/FIPS186 style.
 * This is basically a DSA-specific NSS_PrivateKeyInfo.
 *
 * NSS_DSAPrivateKeyBSAFE.privateKey is an octet string containing
 * the DER encoding of this.
 */
const SecAsn1Template kSecAsn1DSAPrivateKeyOctsTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAPrivateKeyOcts)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyOcts, privateKey)},
    {
        0,
    }};

const SecAsn1Template kSecAsn1DSAPrivateKeyBSAFETemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAPrivateKeyBSAFE)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyBSAFE, version)},
    {SEC_ASN1_INLINE, offsetof(NSS_DSAPrivateKeyBSAFE, dsaAlg), kSecAsn1DSAAlgorithmIdBSAFETemplate},
    {SEC_ASN1_OCTET_STRING, offsetof(NSS_DSAPrivateKeyBSAFE, privateKey)},
    {
        0,
    }};

/*
 * DSA Private Key, PKCS8/SMIME style.
 */
const SecAsn1Template kSecAsn1DSAPrivateKeyPKCS8Template[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSAPrivateKeyPKCS8)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSAPrivateKeyPKCS8, version)},
    {SEC_ASN1_INLINE, offsetof(NSS_DSAPrivateKeyPKCS8, dsaAlg), kSecAsn1DSAAlgorithmIdX509Template},
    {SEC_ASN1_OCTET_STRING, offsetof(NSS_DSAPrivateKeyPKCS8, privateKey)},
    {SEC_ASN1_OPTIONAL | SEC_ASN1_CONSTRUCTED | SEC_ASN1_CONTEXT_SPECIFIC | 0,
     offsetof(NSS_DSAPrivateKeyPKCS8, attributes),
     kSecAsn1SetOfAttributeTemplate},
    {
        0,
    }};

const SecAsn1Template kSecAsn1DSASignatureTemplate[] = {
    {SEC_ASN1_SEQUENCE, 0, NULL, sizeof(NSS_DSASignature)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSASignature, r)},
    {SEC_ASN1_INTEGER, offsetof(NSS_DSASignature, s)},
    {
        0,
    }};
