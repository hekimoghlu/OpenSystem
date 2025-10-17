/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef _BYTEBUFFER_H_
#define _BYTEBUFFER_H_

typedef struct byte_buf {
    size_t  len;
    uint8_t  *bytes;
} byteBufferStruct, *byteBuffer;

void printByteBuffer(byteBuffer bb, char *name);

void printBytes(uint8_t *buff, size_t len, char *name);

byteBuffer
mallocByteBuffer(size_t len);

byteBuffer
hexStringToBytes(const char *inhex);

byteBuffer
hexStringToBytesWithSpaces(char *inhex, int breaks);

static inline byteBuffer
hexStringToBytesIfNotNULL(char *inhex) {
    if(inhex) return hexStringToBytes(inhex);
    return NULL;
}

char
*bytesToHexStringWithSpaces(byteBuffer bb, int breaks);

byteBuffer
bytesToBytes(void *bytes, size_t len);

int
bytesAreEqual(byteBuffer b1, byteBuffer b2);

char
*bytesToHexString(byteBuffer bytes);

byteBuffer
genRandomByteBuffer(size_t minSize, size_t maxSize);

size_t
genRandomSize(size_t minSize, size_t maxSize);

#endif /* _BYTEBUFFER_H_ */
