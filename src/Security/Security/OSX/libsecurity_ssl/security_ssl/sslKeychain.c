/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
 * sslKeychain.c - Apple Keychain routines
 */

#include "ssl.h"
#include "sslContext.h"
#include "sslMemory.h"

#include "sslCrypto.h"
#include <Security/SecBase.h>
#include <Security/SecCertificate.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecIdentity.h>
#include <Security/SecPolicy.h>
#include <Security/SecTrust.h>
#include "utilities/SecCFRelease.h"

#include "sslDebug.h"
#include "sslKeychain.h"
#include <string.h>
#include "utilities/simulatecrash_assert.h"


#include <Security/Security.h>
#include <Security/SecKeyPriv.h>
#if SEC_OS_IPHONE
#include <Security/SecECKey.h>
#endif
#include <AssertMacros.h>
#include <tls_handshake.h>

#if TARGET_OS_IPHONE
#include <Security/oidsalg.h>
#include <Security/SecECKey.h>
#endif

/* Private Key operations */
static
SecAsn1Oid oidForSSLHash(SSL_HashAlgorithm hash)
{
    switch (hash) {
        case tls_hash_algorithm_SHA1:
            return CSSMOID_SHA1WithRSA;
        case tls_hash_algorithm_SHA256:
            return CSSMOID_SHA256WithRSA;
        case tls_hash_algorithm_SHA384:
            return CSSMOID_SHA384WithRSA;
        default:
            break;
    }
    // Internal error
    assert(0);
    // This guarantee failure down the line
    return CSSMOID_MD5WithRSA;
}

static
int mySSLPrivKeyRSA_sign(void *key, tls_hash_algorithm hash, const uint8_t *plaintext, size_t plaintextLen, uint8_t *sig, size_t *sigLen)
{
    SecKeyRef keyRef = key;

    if(hash == tls_hash_algorithm_None) {
        return SecKeyRawSign(keyRef, kSecPaddingPKCS1, plaintext, plaintextLen, sig, sigLen);
    } else {
        SecAsn1AlgId  algId;
        algId.algorithm = oidForSSLHash(hash);
        return SecKeySignDigest(keyRef, &algId, plaintext, plaintextLen, sig, sigLen);
    }
}

static
int mySSLPrivKeyRSA_decrypt(void *key, const uint8_t *ciphertext, size_t ciphertextLen, uint8_t *plaintext, size_t *plaintextLen)
{
    SecKeyRef keyRef = key;

    return SecKeyDecrypt(keyRef, kSecPaddingPKCS1, ciphertext, ciphertextLen, plaintext, plaintextLen);
}

static
int mySSLPrivKeyECDSA_sign(void *key, const uint8_t *plaintext, size_t plaintextLen, uint8_t *sig, size_t *sigLen)
{
    SecKeyRef keyRef = key;

    return SecKeyRawSign(keyRef, kSecPaddingPKCS1, plaintext, plaintextLen, sig, sigLen);
 }

void sslFreePrivKey(tls_private_key_t *sslPrivKey)
{
    assert(sslPrivKey);

    if(*sslPrivKey) {
        CFReleaseSafe(tls_private_key_get_context(*sslPrivKey));
        tls_private_key_destroy(*sslPrivKey);
        *sslPrivKey = NULL;
    }
}

OSStatus
parseIncomingCerts(
	SSLContext			*ctx,
	CFArrayRef			certs,
	SSLCertificate		**destCertChain, /* &ctx->{localCertChain,encryptCertChain} */
	tls_private_key_t   *sslPrivKey)	 /* &ctx->signingPrivKeyRef, etc. */
{
	OSStatus			ortn;
	CFIndex				ix, numCerts;
	SecIdentityRef 		identity;
	SSLCertificate      *certChain = NULL;	/* Retained */
	SecCertificateRef	leafCert = NULL;	/* Retained */
	SecKeyRef           privKey = NULL;	/* Retained */

	assert(ctx != NULL);
	assert(destCertChain != NULL);		/* though its referent may be NULL */
	assert(sslPrivKey != NULL);

	if (certs == NULL) {
		sslErrorLog("parseIncomingCerts: NULL incoming cert array\n");
		ortn = errSSLBadCert;
		goto errOut;
	}
	numCerts = CFArrayGetCount(certs);
	if (numCerts == 0) {
		sslErrorLog("parseIncomingCerts: empty incoming cert array\n");
		ortn = errSSLBadCert;
		goto errOut;
	}

    certChain=sslMalloc(numCerts*sizeof(SSLCertificate));
    if (!certChain) {
        ortn = errSecAllocate;
        goto errOut;
    }

	/*
	 * Certs[0] is an SecIdentityRef from which we extract subject cert,
	 * privKey, pubKey.
	 *
	 * 1. ensure the first element is a SecIdentityRef.
	 */
	identity = (SecIdentityRef)CFArrayGetValueAtIndex(certs, 0);
	if (identity == NULL) {
		sslErrorLog("parseIncomingCerts: bad cert array (1)\n");
		ortn = errSecParam;
		goto errOut;
	}
	if (CFGetTypeID(identity) != SecIdentityGetTypeID()) {
		sslErrorLog("parseIncomingCerts: bad cert array (2)\n");
		ortn = errSecParam;
		goto errOut;
	}

	/*
	 * 2. Extract cert, keys and convert to local format.
	 */
	ortn = SecIdentityCopyCertificate(identity, &leafCert);
	if (ortn) {
		sslErrorLog("parseIncomingCerts: bad cert array (3)\n");
		goto errOut;
	}

	/* Fetch private key from identity */
	ortn = SecIdentityCopyPrivateKey(identity, &privKey);
	if (ortn) {
		sslErrorLog("parseIncomingCerts: SecIdentityCopyPrivateKey err %d\n",
			(int)ortn);
		goto errOut;
	}

    /* Convert the input array of SecIdentityRef at the start to an array of
     all certificates. */
    SSLCopyBufferFromData(SecCertificateGetBytePtr(leafCert), SecCertificateGetLength(leafCert), &certChain[0].derCert);
    certChain[0].next = NULL;

	for (ix = 1; ix < numCerts; ++ix) {
		SecCertificateRef intermediate =
			(SecCertificateRef)CFArrayGetValueAtIndex(certs, ix);
		if (intermediate == NULL) {
			sslErrorLog("parseIncomingCerts: bad cert array (5)\n");
			ortn = errSecParam;
			goto errOut;
		}
		if (CFGetTypeID(intermediate) != SecCertificateGetTypeID()) {
			sslErrorLog("parseIncomingCerts: bad cert array (6)\n");
			ortn = errSecParam;
			goto errOut;
		}

        SSLCopyBufferFromData(SecCertificateGetBytePtr(intermediate), SecCertificateGetLength(intermediate), &certChain[ix].derCert);
        certChain[ix].next = NULL;
        certChain[ix-1].next = &certChain[ix];

	}

    sslFreePrivKey(sslPrivKey);
    size_t size = SecKeyGetBlockSize(privKey);
    if(SecKeyGetAlgorithmId(privKey) == kSecRSAAlgorithmID) {
        *sslPrivKey = tls_private_key_rsa_create(privKey, SecKeyGetBlockSize(privKey), mySSLPrivKeyRSA_sign, mySSLPrivKeyRSA_decrypt);
    } else if (SecKeyGetAlgorithmId(privKey) == kSecECDSAAlgorithmID) {
#if TARGET_OS_IPHONE
        /* Compute signature size from key size */
        size_t sigSize = 8+2*size;
#else
        size_t sigSize = size;
#endif
        *sslPrivKey = tls_private_key_ecdsa_create(privKey, sigSize, SecECKeyGetNamedCurve(privKey), mySSLPrivKeyECDSA_sign);
    } else {
        ortn = errSecParam;
        goto errOut;
    }
    if(*sslPrivKey)
        ortn = errSecSuccess;
    else
        ortn = errSecAllocate;

	/* SUCCESS */
errOut:
	CFReleaseSafe(leafCert);

    sslFree(*destCertChain);

	if (ortn) {
		free(certChain);
		CFReleaseSafe(privKey);
		*destCertChain = NULL;
	} else {
		*destCertChain = certChain;
	}

	return ortn;
}
