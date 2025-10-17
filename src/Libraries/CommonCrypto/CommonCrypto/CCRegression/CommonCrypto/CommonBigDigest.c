/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include "testbyteBuffer.h"
#include "capabilities.h"
#include "testmore.h"
#include <string.h>
#include <CommonCrypto/CommonDigest.h>

#if (CCBIGDIGEST == 0)
entryPoint(CommonBigDigest,"CommonCrypto CCDigest Large Size test")
#else
#include <CommonCrypto/CommonDigestSPI.h>
#include <stdlib.h>

static const size_t blocksz = (size_t) 0x40000000; // 1 GB
static const size_t testsz = (size_t) 0x140000000; // 5 GB

static void DigestInChunks(CCDigestAlgorithm algorithm, size_t chunksize, const uint8_t *bytesToDigest, size_t numbytes, uint8_t *outbuf)
{
    CCDigestRef d = CCDigestCreate(algorithm);
    while(numbytes) {
        size_t n = (numbytes < chunksize) ? numbytes: chunksize;
        CCDigestUpdate(d, bytesToDigest, n);
        numbytes -= n; bytesToDigest += n;
    }
    if(CCDigestFinal(d, outbuf)) return;
    CCDigestDestroy(d);
}

/*
 * Compute the digest of a whole file
 */

static int
checksum_mem(uint8_t *memptr, CCDigestAlgorithm algorithm)
{
    size_t digestsize = CCDigestGetOutputSize(algorithm);
    uint8_t mdwhole[digestsize];
    uint8_t mdchunk[digestsize];
        
	/*
	 * First do it in one big chunk
	 */
    
    CCDigest(algorithm, memptr, testsz, mdwhole);
    
	/*
	 * Now do it in several 1GB chunks
	 */
    
    DigestInChunks(algorithm, blocksz, memptr, testsz, mdchunk);
    
    int cmpval = memcmp(mdchunk, mdwhole, digestsize);
    ok(cmpval == 0, "Results are the same for both digests");
    
    return 0;
}

static inline uint8_t *
getMemoryPattern() {
    uint8_t *retval = malloc(testsz);
    if(!retval) return retval;
    for(size_t i=0; i<testsz; i++) retval[i] = i & 0xff;
    return retval;
}

static const int kTestTestCount = 1;

int CommonBigDigest(int __unused argc, char *const * __unused argv)
{
    plan_tests(kTestTestCount);
    uint8_t *memptr = getMemoryPattern();
    if(memptr) {
        checksum_mem(memptr, kCCDigestSHA1);
        free(memptr);
    } else {
        ok(1, "can't get enough memory");
        diag("Can't check large digest correctness - failed to alloc memory\n");
    }
    return 0;
}


#endif
