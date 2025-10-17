/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#import "aks.h"
#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <Security/SecRandom.h>
#import <CommonCrypto/CommonCryptor.h>
#import <CommonCrypto/CommonCryptorSPI.h>
#import <TargetConditionals.h>
#import "gssoslog.h"
#import "heimbase.h"
#import <os/transaction_private.h>

#define PLATFORM_SUPPORT_CLASS_F !TARGET_OS_SIMULATOR

#import <AssertMacros.h>
#if PLATFORM_SUPPORT_CLASS_F
#import <libaks.h>
#endif
#import "HeimCredCoder.h"
#import "common.h"
#import "roken.h"

/*
 * stored as [32:wrapped_key_len][wrapped_key_len:wrapped_key][iv:ivSize][variable:ctdata][16:tag]
 */

static const size_t ivSize = 16;
#if PLATFORM_SUPPORT_CLASS_F
static os_transaction_t keyNotReadyTransaction = NULL;
#endif

NSData *
ksEncryptData(NSData *plainText)
{
    NSMutableData *blob = NULL;
    
    const size_t bulkKeySize = 32; /* Use 256 bit AES key for bulkKey. */
    const uint32_t maxKeyWrapOverHead = 8 + 32;
    uint8_t bulkKey[bulkKeySize];
    uint8_t iv[ivSize];
    uint8_t bulkKeyWrapped[bulkKeySize + maxKeyWrapOverHead];
    uint32_t key_wrapped_size;
    CCCryptorStatus ccerr;

    heim_assert([plainText isKindOfClass:[NSData class]], "input is not NSData");
    
    size_t ctLen = [plainText length];
    size_t tagLen = 16;

    if (SecRandomCopyBytes(kSecRandomDefault, bulkKeySize, bulkKey)) {
	abort();
    }
    if (SecRandomCopyBytes(kSecRandomDefault, ivSize, iv)) {
	abort();
    }

    int bulkKeyWrappedSize;
#if PLATFORM_SUPPORT_CLASS_F
    kern_return_t error;

    bulkKeyWrappedSize = sizeof(bulkKeyWrapped);

    error = aks_wrap_key(bulkKey, sizeof(bulkKey), key_class_f, bad_keybag_handle, bulkKeyWrapped, &bulkKeyWrappedSize, NULL);
    if (error) {
	os_log_error(GSSOSLog(), "Error with wrap key: %d", error);
	// When there is a key error, start an os transaction instead of aborting.  This will keep the service running for the users. The risk is that if the service exits, then all the credentials are lost.  While not ideal, it is better than the service crashing when it tries to save the credentials.
	if (!keyNotReadyTransaction) {
	    keyNotReadyTransaction = os_transaction_create("com.apple.Heimdal.GSSCred.keyError");
	}
	return NULL;
    }
    //complete the transaction, if present
    if (keyNotReadyTransaction) {
	keyNotReadyTransaction = NULL;
    }
    if ((unsigned long)bulkKeyWrappedSize > sizeof(bulkKeyWrapped)) {
	abort();
    }

#else
    bulkKeyWrappedSize = bulkKeySize;
    memcpy(bulkKeyWrapped, bulkKey, bulkKeySize);
#endif
    key_wrapped_size = (uint32_t)bulkKeyWrappedSize;
    
    size_t blobLen = sizeof(key_wrapped_size) + key_wrapped_size + ivSize + ctLen + tagLen;
    
    blob = [[NSMutableData alloc] initWithLength:blobLen];
    if (blob == NULL) {
	return NULL;
    }

    UInt8 *cursor = [blob mutableBytes];


    memcpy(cursor, &key_wrapped_size, sizeof(key_wrapped_size));
    cursor += sizeof(key_wrapped_size);
    
    memcpy(cursor, bulkKeyWrapped, key_wrapped_size);
    cursor += key_wrapped_size;

    memcpy(cursor, iv, ivSize);
    cursor += ivSize;

    ccerr = CCCryptorGCMOneshotEncrypt(kCCAlgorithmAES,       // algorithm
				       bulkKey,               // key bytes
				       bulkKeySize,           // key length
				       iv,                    // IV/nonce bytes
				       ivSize,                // IV/nonce length
				       NULL,                  // additional bytes
				       0,                     // additional bytes length
				       plainText.bytes,       // plaintext bytes
				       ctLen,                 // plaintext length
				       cursor,                // ciphertext bytes
				       cursor + ctLen,        // authentication tag bytes
				       tagLen);               // authentication tag length
    memset_s(bulkKey, 0, sizeof(bulkKey), sizeof(bulkKey));
    if (ccerr || tagLen != 16) {
	return NULL;
    }

    return blob;
}

NSData *
ksDecryptData(NSData * blob)
{
    const uint32_t bulkKeySize = 32; /* Use 256 bit AES key for bulkKey. */
    uint8_t bulkKey[bulkKeySize];
    int error = EINVAL;
    CCCryptorStatus ccerr;
    uint8_t *tag = NULL;
    const uint8_t *iv = NULL;
    NSMutableData *clear = NULL, *plainText = NULL;

    size_t blobLen = [blob length];
    const uint8_t *cursor = [blob bytes];

    uint32_t wrapped_key_size;
    
    size_t ctLen = blobLen;
    
    /* tag is stored after the plain text data */
    size_t tagLen = 16;
    if (ctLen < tagLen)
	return NULL;
    ctLen -= tagLen;

    if (ctLen < sizeof(wrapped_key_size))
	return NULL;

    memcpy(&wrapped_key_size, cursor, sizeof(wrapped_key_size));

    cursor += sizeof(wrapped_key_size);
    ctLen -= sizeof(wrapped_key_size);

    /* Validate key wrap length against total length */
    if (ctLen < wrapped_key_size)
	return NULL;

    int keySize = sizeof(bulkKey);
#if PLATFORM_SUPPORT_CLASS_F

    error = aks_unwrap_key(cursor, wrapped_key_size, key_class_f, bad_keybag_handle, bulkKey, &keySize);
    if (error != KERN_SUCCESS) {
	os_log_error(GSSOSLog(), "Error with unwrap key: %d", error);
	goto out;
    }
#else
    if (bulkKeySize != wrapped_key_size) {
	error = EINVAL;
	goto out;
    }
    memcpy(bulkKey, cursor, bulkKeySize);
    keySize = 32;
#endif

    if (keySize != 32) {
	error = EINVAL;
	goto out;
    }

    cursor += wrapped_key_size;
    ctLen -= wrapped_key_size;

    if (ctLen < ivSize) {
	error = EINVAL;
	goto out;
    }

    iv = cursor;
    cursor += ivSize;
    ctLen -= ivSize;

    plainText = [NSMutableData dataWithLength:ctLen];
    if (!plainText) {
        goto out;
    }

    tag = malloc(tagLen);
    if (tag == NULL) {
        goto out;
    }

    ccerr = CCCryptorGCMOneshotDecrypt(kCCAlgorithmAES,           // algorithm
				       bulkKey,                   // key bytes
				       bulkKeySize,               // key length
				       iv,                        // IV/nonce bytes
				       ivSize,                    // IV/nonce length
				       NULL,                      // additional bytes
				       0,                         // additional bytes length
				       cursor,                    // ciphertext bytes
				       ctLen,                     // ciphertext length
				       plainText.mutableBytes,    // plaintext bytes
				       cursor + ctLen,            // authentication tag bytes
				       tagLen);                   // authentication tag length
    /* Decrypt the cipherText with the bulkKey. */
    if (ccerr) {
	goto out;
    }

    clear = plainText;
out:
    memset_s(bulkKey, 0, bulkKeySize, bulkKeySize);
    free(tag);

    return clear;
}

