/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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

// ZLIB tests tools
// CM 2022/08/29
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "zlib.h"

#define PLOG(F, ...)  do { fprintf(stderr, F"\n", ##__VA_ARGS__); } while (0)
#define PFAIL(F, ...) do { PLOG("[ERR "__FILE__":%s:%d] "F, __FUNCTION__, __LINE__, ##__VA_ARGS__); exit(1); } while (0)

#pragma mark - BENCHMARKING

int kpc_cycles_init(void);
uint64_t kpc_get_cycles(void);

#pragma mark - BUFFER API

// Encode buffer using zlib. Return number of compressed bytes, 0 on failure.
size_t zlib_encode_buffer(uint8_t* dst_buffer, size_t dst_size,
                          uint8_t* src_buffer, size_t src_size, int level, int rfc1950, int fixed);

// Decode buffer using zlib. Return number of uncompressed bytes, 0 on failure.
// Supports truncated decodes.
size_t zlib_decode_buffer(uint8_t* dst_buffer, size_t dst_size,
                          uint8_t* src_buffer, size_t src_size, int rfc1950);

// Decode buffer using zlib using infback interface. Return number of uncompressed bytes, 0 on failure.
// Supports truncated decodes.
size_t zlib_decode_infback(uint8_t* dst_buffer, size_t dst_size,
                           uint8_t* src_buffer, size_t src_size);

// Decode buffer using zlib. Torture streaming API. Return number of uncompressed bytes, 0 on failure.
size_t zlib_decode_torture(uint8_t* dst_buffer, size_t dst_size,
                           uint8_t* src_buffer, size_t src_size, int rfc1950);

#pragma mark - CHECKSUMS

// Return Crc32 of DATA[LEN]. Naive implementation.
uint32_t simple_crc32(uint8_t* src_buffer, const size_t src_size);

// Return Adler32 of DATA[LEN]. Naive implementation.
uint32_t simple_adler32(const unsigned char* src, const size_t src_size);
