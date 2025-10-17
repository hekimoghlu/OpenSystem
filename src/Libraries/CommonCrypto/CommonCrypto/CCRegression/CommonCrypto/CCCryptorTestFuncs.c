/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include <stdio.h>
#include <assert.h>
#include "CCCryptorTestFuncs.h"
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"
#include <CommonCrypto/CommonCryptorSPI.h>

CCCryptorStatus
CCCryptWithMode(CCOperation op, CCMode mode, CCAlgorithm alg, CCPadding padding, const void *iv, 
				const void *key, size_t keyLength, const void *tweak, size_t tweakLength,
                int numRounds, CCModeOptions options,
                const void *dataIn, size_t dataInLength, 
                void *dataOut, size_t dataOutAvailable, size_t *dataOutMoved)
#ifdef CRYPTORWITHMODE
{
    CCCryptorRef cref;
	CCCryptorStatus retval;
    size_t moved;

   	if((retval = CCCryptorCreateWithMode(op, mode, alg, padding, iv, key, keyLength, tweak, tweakLength, numRounds, options, &cref)) != kCCSuccess) {
    	return retval;
    }
    
    if((retval = CCCryptorUpdate(cref, dataIn, dataInLength, dataOut, dataOutAvailable, &moved)) != kCCSuccess) {
    	return retval;
    }
    
    dataOut += moved;
    dataOutAvailable -= moved;
    *dataOutMoved = moved;
    
    if((retval = CCCryptorFinal(cref, dataOut, dataOutAvailable, &moved)) != kCCSuccess) {
    	return retval;
    }
    
    *dataOutMoved += moved;

	CCCryptorRelease(cref);
    
    return kCCSuccess;
}
#else
{
    return kCCSuccess;
}
#endif



CCCryptorStatus 
CCMultiCrypt(CCOperation op, CCAlgorithm alg, CCOptions options, const void *key, size_t keyLength, const void *iv, const void *dataIn, size_t dataInLength,
	void *dataOut, size_t dataOutAvailable, size_t *dataOutMoved)
{
	CCCryptorRef cref;
    CCCryptorStatus retval;
    size_t p1, p2;
    size_t newmoved;
    size_t finalSize;
    
    retval = CCCryptorCreate(op, alg, options, key, keyLength, iv, &cref);
    if(retval != kCCSuccess) {
    	diag("Cryptor Create Failed\n");
    	return retval;
    }
    p1 = ( dataInLength / 16 ) * 16 - 1;
    if(p1 > 16) p1 = dataInLength;
    p2 = dataInLength - p1;
    // diag("Processing length %d  in two parts %d and %d\n", (int) dataInLength, (int) p1, (int) p2);
    
    *dataOutMoved = 0;
    
    if(p1) {
    	retval = CCCryptorUpdate(cref, dataIn, p1, dataOut, dataOutAvailable, dataOutMoved);
        if(retval) {
        	diag("P1 - Tried to move %d - failed retval = %d\n", (int) p1, (int) retval);
            return retval;
        }
        dataIn += p1;
        dataOut += *dataOutMoved;
        dataOutAvailable -= *dataOutMoved;        
    }
    if(p2) {
        
    	retval = CCCryptorUpdate(cref, dataIn, p2, dataOut, dataOutAvailable, &newmoved);
        if(retval) {
        	diag("P2 - Tried to move %d - failed\n", (int) p2);
            return retval;
        }
        dataOut += newmoved;        
        dataOutAvailable -= newmoved;
        *dataOutMoved += newmoved;
    }
    
    /* We've had reports that Final fails on some platforms if it's only cipher blocksize.  */
    switch(alg) {
    case kCCAlgorithmDES: /* fallthrough */
    case kCCAlgorithm3DES: finalSize = kCCBlockSizeDES; break;
    case kCCAlgorithmAES128: finalSize = kCCBlockSizeAES128; break;
    case kCCAlgorithmCAST: finalSize = kCCBlockSizeCAST; break;
    case kCCAlgorithmRC2: finalSize = kCCBlockSizeRC2; break;
    default: finalSize = dataOutAvailable;
    }
    
    retval = CCCryptorFinal(cref, dataOut, finalSize, &newmoved);
    if(retval) {
        diag("Final - failed %d\n", (int) retval);
        return retval;
    }
    retval = CCCryptorRelease(cref);
    if(retval) {
        diag("Final - release failed %d\n", (int) retval);
        return retval;
    }
    *dataOutMoved += newmoved;
    return kCCSuccess;
    
    
}

CCCryptorStatus 
CCMultiCryptWithMode(CCOperation op, CCMode mode, CCAlgorithm alg, CCPadding padding, const void *iv, 
	const void *key, size_t keyLength, const void *tweak, size_t tweakLength,
	int numRounds, CCModeOptions options,
    const void *dataIn, size_t dataInLength,
	void *dataOut, size_t dataOutAvailable, size_t *dataOutMoved)
#ifdef CRYPTORWITHMODE
{
	CCCryptorRef cref;
    CCCryptorStatus retval;
    size_t p1, p2;
    size_t newmoved;
    
    retval = CCCryptorCreateWithMode(op, mode, alg, padding, iv, key, keyLength, tweak, tweakLength, numRounds, options, &cref);
   	if(retval != kCCSuccess) {
    	return retval;
    }
    p1 = ( dataInLength / 16 ) * 16 - 1;
    if(p1 > 16) p1 = dataInLength;
    p2 = dataInLength - p1;
    // diag("Processing length %d  in two parts %d and %d\n", (int) dataInLength, (int) p1, (int) p2);
    
    *dataOutMoved = 0;
    
    if(p1) {
    	retval = CCCryptorUpdate(cref, dataIn, p1, dataOut, dataOutAvailable, dataOutMoved);
        if(retval) {
        	diag("P1 - Tried to move %d - failed retval = %d\n", (int) p1, (int) retval);
            return retval;
        }
        dataIn += p1;
        dataOut += *dataOutMoved;
        dataOutAvailable -= *dataOutMoved;        
    }
    if(p2) {
        
    	retval = CCCryptorUpdate(cref, dataIn, p2, dataOut, dataOutAvailable, &newmoved);
        if(retval) {
        	diag("P2 - Tried to move %d - failed\n", (int) p2);
            return retval;
        }
        dataOut += newmoved;        
        dataOutAvailable -= newmoved;
        *dataOutMoved += newmoved;
    }
    retval = CCCryptorFinal(cref, dataOut, dataOutAvailable, &newmoved);
    if(retval) {
        diag("Final - failed %d\n", (int) retval);
        return retval;
    }
    retval = CCCryptorRelease(cref);
    if(retval) {
        diag("Final - release failed %d\n", (int) retval);
        return retval;
    }
    *dataOutMoved += newmoved;
    return kCCSuccess;
}
#else
{
    return kCCSuccess;
}
#endif


byteBuffer
ccConditionalTextBuffer(char *inputText)
{
	byteBuffer ret;
    
    if(inputText) ret = hexStringToBytes(inputText);
    else {
    	ret = hexStringToBytes("");
        ret->bytes = NULL;
    }
    return ret;
}

#define log(do_print, MSG, ARGS...) \
if(do_print){test_diag(test_directive, test_reason, __FILE__, __LINE__, MSG, ## ARGS);}

int
CCCryptTestCase(char *keyStr, char *ivStr, CCAlgorithm alg, CCOptions options, char *cipherText, char *plainText, bool log)
{
    byteBuffer key, iv;
    byteBuffer pt, ct;
    byteBuffer bb=NULL, bb2=NULL;
    int rc=1; //error

	CCCryptorStatus retval;
    char cipherDataOut[4096];
    char plainDataOut[4096];
    size_t dataOutMoved;

            
    key = hexStringToBytes(keyStr);        
    pt = ccConditionalTextBuffer(plainText);
    ct = ccConditionalTextBuffer(cipherText);
    iv = ccConditionalTextBuffer(ivStr);

    if (alg==kCCAlgorithmAES) {
        //feed a wrong key length
        retval = CCCrypt(kCCEncrypt, alg, options, key->bytes, key->len-2, iv->bytes, pt->bytes, pt->len, cipherDataOut, 4096, &dataOutMoved);
        if (retval!=kCCKeySizeError)
            goto errOut;
    }
    
    retval = CCCrypt(kCCEncrypt, alg, options, key->bytes, key->len, iv->bytes, pt->bytes, pt->len, cipherDataOut, 4096, &dataOutMoved);
    if(retval != kCCSuccess) {
    	log(log, "Encrypt Failed %d\n", retval);
        goto errOut;
    }
    
    bb = bytesToBytes(cipherDataOut, dataOutMoved);

    // If ct isn't defined we're gathering data - print the ciphertext result
    if(!ct->bytes) {
    	log(log, "Input Length %d Result: %s\n", (int) pt->len, bytesToHexString(bb));
    } else {
        if (!bytesAreEqual(ct, bb)) {
            log(log, "FAIL Encrypt Output %s\nEncrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(ct));
        	goto errOut;
        }
    }

    retval = CCCrypt(kCCDecrypt, alg, options, key->bytes, key->len, iv->bytes, cipherDataOut, dataOutMoved, plainDataOut, 4096, &dataOutMoved);
    if(retval != kCCSuccess) {
    	log(log, "Decrypt Failed\n");
        goto errOut;
    }
    
    bb2 = bytesToBytes(plainDataOut, dataOutMoved);
    
	if (!bytesAreEqual(pt, bb2)) {
        log(log, "FAIL Decrypt Output %s\nDecrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(pt));
        goto errOut;
    }

    rc=0;

errOut:
    free(bb2);
    free(bb);
    free(pt);
    free(ct);
    free(key);
    free(iv);
	return rc;
}

int
CCMultiCryptTestCase(char *keyStr, char *ivStr, CCAlgorithm alg, CCOptions options, char *cipherText, char *plainText)
{
    byteBuffer key, iv;
    byteBuffer pt, ct;
    
    
	CCCryptorStatus retval;
    char cipherDataOut[4096];
    char plainDataOut[4096];
    size_t dataOutMoved;
    byteBuffer bb;
            
    key = hexStringToBytes(keyStr);        
    pt = ccConditionalTextBuffer(plainText);
    ct = ccConditionalTextBuffer(cipherText);
    iv = ccConditionalTextBuffer(ivStr);
    
    retval = CCMultiCrypt(kCCEncrypt, alg, options, key->bytes, key->len, iv->bytes, pt->bytes, pt->len, cipherDataOut, 4096, &dataOutMoved);
    if(retval != kCCSuccess) {
    	diag("Encrypt Failed\n");
        return 1;
    }
    
    bb = bytesToBytes(cipherDataOut, dataOutMoved);    	

    // If ct isn't defined we're gathering data - print the ciphertext result
    if(!ct->bytes) {
    	diag("Input Length %d Result: %s\n", (int) pt->len, bytesToHexString(bb));
    } else {
        if (!bytesAreEqual(ct, bb)) {
            diag("FAIL Encrypt Output %s\nEncrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(ct));
        	return 1;
        }
    }
    
    free(bb);
    
    retval = CCMultiCrypt(kCCDecrypt, alg, options, key->bytes, key->len, iv->bytes, cipherDataOut, dataOutMoved, plainDataOut, 4096, &dataOutMoved);
    if(retval != kCCSuccess) {
    	diag("Decrypt Failed\n");
        return 1;
    }
    
    bb = bytesToBytes(plainDataOut, dataOutMoved);
    
	if (!bytesAreEqual(pt, bb)) {
        diag("FAIL Decrypt Output %s\nDecrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(pt));
        return 1;
    }

    free(bb);
    free(pt);
    free(ct);
    free(key);
    free(iv);
	return 0;
}




int
CCModeTestCase(char *keyStr, char *ivStr, CCMode mode, CCAlgorithm alg, CCPadding padding, char *cipherText, char *plainText)
#ifdef CRYPTORWITHMODE
{
    byteBuffer key, iv;
    byteBuffer pt, ct;
    
	CCCryptorStatus retval;
    char cipherDataOut[4096];
    char plainDataOut[4096];
    size_t dataOutMoved;
    byteBuffer bb;
            
    key = hexStringToBytes(keyStr);        
    pt = ccConditionalTextBuffer(plainText);
    ct = ccConditionalTextBuffer(cipherText);
    iv = ccConditionalTextBuffer(ivStr);
    
    retval = CCCryptWithMode(kCCEncrypt, mode, alg, padding, iv->bytes, key->bytes, key->len, NULL, 0, 0, 0,  pt->bytes, pt->len,
                             cipherDataOut, 4096, &dataOutMoved);
   	if(retval != kCCSuccess) {
    	diag("Encrypt Failed\n");
        return 1;
    }
    
    bb = bytesToBytes(cipherDataOut, dataOutMoved);    	

    // If ct isn't defined we're gathering data - print the ciphertext result
    if(!ct->bytes) {
    	diag("Input Length %d Result: %s\n", (int) pt->len, bytesToHexString(bb));
    } else {
        if (!bytesAreEqual(ct, bb)) {
            diag("FAIL\nEncrypt Output %s\nEncrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(ct));
        	return 1;
        }
    }
    
    free(bb);
    
    retval = CCCryptWithMode(kCCDecrypt, mode, alg, padding, iv->bytes, key->bytes, key->len, NULL, 0, 0, 0,  cipherDataOut, dataOutMoved,
                             plainDataOut, 4096, &dataOutMoved);
   	if(retval != kCCSuccess) {
    	diag("Decrypt Failed\n");
        return 1;
    }
    
    bb = bytesToBytes(plainDataOut, dataOutMoved);
    
	if (!bytesAreEqual(pt, bb)) {
        diag("FAIL Decrypt Output %s\nDecrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(pt));
        return 1;
    }

    free(bb);
    free(pt);
    free(ct);
    free(key);
    free(iv);
	return 0;
}
#else
{
    return 0;
}
#endif




int
CCMultiModeTestCase(char *keyStr, char *ivStr, CCMode mode, CCAlgorithm alg, CCPadding padding, char *cipherText, char *plainText)
#ifdef CRYPTORWITHMODE
{
    byteBuffer key, iv;
    byteBuffer pt, ct;    
	CCCryptorStatus retval;
    char cipherDataOut[4096];
    char plainDataOut[4096];
    size_t dataOutMoved;
    byteBuffer bb;
            
    key = hexStringToBytes(keyStr);        
    pt = ccConditionalTextBuffer(plainText);
    ct = ccConditionalTextBuffer(cipherText);
    iv = ccConditionalTextBuffer(ivStr);
    
    retval = CCMultiCryptWithMode(kCCEncrypt, mode, alg, padding, iv->bytes,key->bytes, key->len, NULL, 0,0, 0, pt->bytes, pt->len,
                                  cipherDataOut, 4096, &dataOutMoved);
   	if(retval != kCCSuccess) {
    	diag("Encrypt Failed\n");
        return 1;
    }
    
    bb = bytesToBytes(cipherDataOut, dataOutMoved);    	

    // If ct isn't defined we're gathering data - print the ciphertext result
    if(!ct->bytes) {
    	diag("Input Length %d Result: %s\n", (int) pt->len, bytesToHexString(bb));
    } else {
        if (!bytesAreEqual(ct, bb)) {
            diag("FAIL\nEncrypt Output %s\nEncrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(ct));
        	return 1;
        }
    }
    
    free(bb);
    
    retval = CCMultiCryptWithMode(kCCEncrypt, mode, alg, padding, iv->bytes, key->bytes, key->len, NULL, 0, 0, 0,
                                  cipherDataOut, dataOutMoved, plainDataOut, 4096, &dataOutMoved);
   	if(retval != kCCSuccess) {
    	diag("Decrypt Failed\n");
        return 1;
    }
    
    bb = bytesToBytes(plainDataOut, dataOutMoved);
    
	if (!bytesAreEqual(pt, bb)) {
        diag("FAIL Decrypt Output %s\nDecrypt Expect %s\n", bytesToHexString(bb), bytesToHexString(pt));
        return 1;
    }

    free(bb);
    free(pt);
    free(ct);
    free(key);
    free(iv);
	return 0;
}
#else
{
    return kCCSuccess;
}
#endif

