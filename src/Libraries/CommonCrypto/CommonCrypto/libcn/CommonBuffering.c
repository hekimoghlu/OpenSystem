/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

//
//  CommonBuffering.c
//  CommonCrypto
//

#include <stdlib.h>
#include <stdio.h>
#include <corecrypto/cc.h>
#include "CommonBufferingPriv.h"
#include "../lib/cc_macros_priv.h"

CNBufferRef
CNBufferCreate(size_t chunksize)
{
    CNBufferRef retval = malloc(sizeof(CNBuffer));
    __Require_Quiet(NULL != retval, errOut);
    retval->chunksize = chunksize;
    retval->bufferPos = 0;
    retval->buf = malloc(chunksize);
    __Require_Quiet(NULL != retval->buf, errOut);
    return retval;
    
errOut:
    if(retval) {
        if(retval->buf) free(retval->buf);
        free(retval);
    }
    return NULL;
}

CNStatus
CNBufferRelease(CNBufferRef *bufRef)
{
    CNBufferRef ref;
    
    __Require_Quiet(NULL != bufRef, out);

    ref = *bufRef;
    if(ref->buf) free(ref->buf);
    if(ref) free(ref);
out:
    return kCNSuccess;
}



CNStatus
CNBufferProcessData(CNBufferRef bufRef, 
                    void *ctx, const void *in, const size_t inLen, void *out, size_t *outLen, 
                    cnProcessFunction pFunc, cnSizeFunction sizeFunc)
{
    size_t  blocksize = bufRef->chunksize;
    const uint8_t *input = in;
    uint8_t *output = out;
    size_t inputLen = inLen, outputLen, inputUsing, outputAvailable;
    
    outputAvailable = outputLen = *outLen;
    
    if(sizeFunc(ctx, bufRef->bufferPos + inLen) > outputAvailable) return  kCNBufferTooSmall;
    *outLen = 0;
    if(bufRef->bufferPos > 0) {
        inputUsing = CC_MIN(blocksize - bufRef->bufferPos, inputLen);
        memcpy(&bufRef->buf[bufRef->bufferPos], in, inputUsing);
        bufRef->bufferPos += inputUsing;
        if(bufRef->bufferPos < blocksize) {
            return kCNSuccess;
        }
        pFunc(ctx, bufRef->buf, blocksize, output, &outputLen);
        inputLen -= inputUsing; input += inputUsing;
        output += outputLen; *outLen = outputLen; outputAvailable -= outputLen;
        bufRef->bufferPos = 0;
    }
    
    inputUsing = inputLen - inputLen % blocksize;
    if(inputUsing > 0) {
        outputLen = outputAvailable;
        pFunc(ctx, input, inputUsing, output, &outputLen);
        inputLen -= inputUsing; input += inputUsing;
        *outLen += outputLen;
    }
    
    if(inputLen > blocksize) {
        return kCNAlignmentError;
    } else if(inputLen > 0) {
        memcpy(bufRef->buf, input, inputLen);
        bufRef->bufferPos = inputLen;
    }
    return kCNSuccess;
    
}

CNStatus
CNBufferFlushData(CNBufferRef bufRef,
                  void *ctx, void *out, size_t *outLen,
                  cnProcessFunction pFunc, cnSizeFunction sizeFunc)
{
//    size_t outputLen, outputAvailable;
//    outputAvailable = outputLen = *outLen;

    if(bufRef->bufferPos > 0) {
        if(bufRef->bufferPos > bufRef->chunksize) return kCNAlignmentError;
        if(sizeFunc(ctx, bufRef->bufferPos) > *outLen) return kCNBufferTooSmall;
        pFunc(ctx, bufRef->buf, bufRef->bufferPos, out, outLen);
    } else {
        *outLen = 0;
    }
    return kCNSuccess;
}



bool
CNBufferEmpty(CNBufferRef bufRef)
{
    return bufRef->bufferPos == 0;
}
