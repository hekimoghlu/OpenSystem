/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
 * CommonCryptorPriv.h - interface between CommonCryptor and operation- and
 *           algorithm-specific service providers. 
 */

#ifndef _CC_COMMON_CRYPTOR_PRIV_
#define _CC_COMMON_CRYPTOR_PRIV_

#include  <CommonCrypto/CommonCryptor.h>
#include  <CommonCrypto/CommonCryptorSPI.h>
#include "ccDispatch.h"

#include "corecryptoSymmetricBridge.h"

#ifdef DEBUG
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
    
    /* Byte-Size Constants */
#define CCMAXBUFFERSIZE 128             /* RC2/RC5 Max blocksize */
#define DEFAULT_CRYPTOR_MALLOC 4096
#define CC_STREAMKEYSCHED  2048
#define CC_MODEKEYSCHED  2048
#define CC_MAXBLOCKSIZE  128

#define ACTIVE 1
#define RELEASED 0xDEADBEEF
    
typedef struct _CCCryptor {
    struct _CCCryptor *compat;
#ifdef DEBUG
    uint64_t        active;
    uint64_t        cryptorID;
#endif
    uint8_t         buffptr[32];
    size_t          bufferPos;
    size_t          bytesProcessed;
    size_t          cipherBlocksize;

    CCAlgorithm     cipher;
    CCMode          mode;
    CCOperation     op;        /* kCCEncrypt, kCCDecrypt, or kCCBoth */
    
    corecryptoMode  symMode[CC_DIRECTIONS];
    const cc2CCModeDescriptor *modeDesc;
    modeCtx         ctx[CC_DIRECTIONS];
    const cc2CCPaddingDescriptor *padptr;
    
} CCCryptor;
    
static inline CCCryptor *
getRealCryptor(CCCryptorRef p, int checkactive) {
    if(!p) return NULL;
    if(p->compat) p = p->compat;
#ifdef DEBUG
    if(checkactive && p->active != ACTIVE) printf("Using Finalized Cryptor %16llx\n", p->cryptorID);
#else
    (void) checkactive;
#endif
    return p;
}
    
#define CCCRYPTOR_SIZE  sizeof(struct _CCCryptor)
#define kCCContextSizeGENERIC (sizeof(struct _CCCryptor))
#define CC_COMPAT_SIZE (sizeof(void *)*2)
    
#define AESGCM_MIN_TAG_LEN 8
#define AESGCM_MIN_IV_LEN  12
#define AESGCM_BLOCK_LEN  16
    
const corecryptoMode getCipherMode(CCAlgorithm cipher, CCMode mode, CCOperation direction);
    
#ifdef __cplusplus
}
#endif

#endif  /* _CC_COMMON_CRYPTOR_PRIV_ */
