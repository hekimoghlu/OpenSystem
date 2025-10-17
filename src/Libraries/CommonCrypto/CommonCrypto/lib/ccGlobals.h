/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
 *  ccGlobals.h - CommonCrypto global DATA
 */

#ifndef CCGLOBALS_H
#define CCGLOBALS_H

#if defined(_MSC_VER) || defined(__ANDROID__)
#else
#include <os/log.h>
#include <os/assumes.h>
#endif

#include <corecrypto/ccdh.h>
#include <corecrypto/ccdigest.h>
#include "CommonCryptorPriv.h"
#include "basexx.h"

#include "crc.h"
#include <CommonNumerics/CommonCRC.h>
#include <CommonCrypto/CommonDigestSPI.h>

#if defined(_WIN32)
    #define _LIBCOMMONCRYPTO_HAS_ALLOC_ONCE 0
#elif __has_include(<os/alloc_once_private.h>)
        #include <os/alloc_once_private.h>
        #if defined(OS_ALLOC_ONCE_KEY_LIBCOMMONCRYPTO)
            #define _LIBCOMMONCRYPTO_HAS_ALLOC_ONCE 1
        #endif
#else
    #define _LIBCOMMONCRYPTO_HAS_ALLOC_ONCE 0
#endif

#define CN_SUPPORTED_CRCS kCN_CRC_64_ECMA_182+1
#define CN_STANDARD_BASE_ENCODERS kCNEncodingBase16+1

#define  CC_MAX_N_DIGESTS (kCCDigestMax)

struct cc_globals_s {
    crcInfo crcSelectionTab[CN_SUPPORTED_CRCS]; // CommonCRC.c
    BaseEncoderFrame encoderTab[CN_STANDARD_BASE_ENCODERS];
	const struct ccdigest_info *digest_info[CC_MAX_N_DIGESTS];// CommonDigest.c
};

typedef struct cc_globals_s *cc_globals_t;
void init_globals(void *g);

static inline cc_globals_t
_cc_globals(void) {
#if _LIBCOMMONCRYPTO_HAS_ALLOC_ONCE
    cc_globals_t globals =  (cc_globals_t) os_alloc_once(OS_ALLOC_ONCE_KEY_LIBCOMMONCRYPTO,
                                        sizeof(struct cc_globals_s),
                                        init_globals);
   if(OS_EXPECT(globals==NULL, 0)){
        struct _os_alloc_once_s *slot = &_os_alloc_once_table[OS_ALLOC_ONCE_KEY_LIBCOMMONCRYPTO-1];
        os_log_fault(OS_LOG_DEFAULT, "slot=%p once=%li, ptr=%p", slot, slot->once, slot->ptr);
        slot = &_os_alloc_once_table[OS_ALLOC_ONCE_KEY_LIBCOMMONCRYPTO];
        os_log_fault(OS_LOG_DEFAULT, "slot=%p once=%li, ptr=%p", slot, slot->once, slot->ptr);
        slot = &_os_alloc_once_table[OS_ALLOC_ONCE_KEY_LIBCOMMONCRYPTO+1];
        os_log_fault(OS_LOG_DEFAULT, "slot=%p once=%li, ptr=%p", slot, slot->once, slot->ptr);

        os_crash("output of os_alloc_once() is NULL");
    }
    return globals;
#else
    extern dispatch_once_t cc_globals_init;
    extern struct cc_globals_s cc_globals_storage;    
    cc_dispatch_once(&cc_globals_init, &cc_globals_storage, init_globals);
    return &cc_globals_storage;
#endif
}

#endif /* CCGLOBALS_H */
