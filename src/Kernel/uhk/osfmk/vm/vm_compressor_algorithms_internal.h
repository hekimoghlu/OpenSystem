/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#ifndef _VM_COMPRESSOR_ALGORITHMS_INTERNAL_H_
#define _VM_COMPRESSOR_ALGORITHMS_INTERNAL_H_

#ifdef XNU_KERNEL_PRIVATE
#include <vm/vm_compressor_algorithms_xnu.h>

int metacompressor(const uint8_t *in, uint8_t *cdst, int32_t outbufsz,
    uint16_t *codec, void *cscratch, boolean_t *, uint32_t *pop_count_p);
bool metadecompressor(const uint8_t *source, uint8_t *dest, uint32_t csize,
    uint16_t ccodec, void *compressor_dscratch, uint32_t *pop_count_p);

typedef enum {
	CCWK = 0, // must be 0 or 1
	CCLZ4 = 1, //must be 0 or 1
	CINVALID = 0xFFFF
} vm_compressor_codec_t;

typedef enum {
	CMODE_WK = 0,
	CMODE_LZ4 = 1,
	CMODE_HYB = 2,
	VM_COMPRESSOR_DEFAULT_CODEC = 3,
	CMODE_INVALID = 4
} vm_compressor_mode_t;

void vm_compressor_algorithm_init(void);
int vm_compressor_algorithm(void);

#endif /* XNU_KERNEL_PRIVATE */
#endif  /* _VM_COMPRESSOR_ALGORITHMS_INTERNAL_H_ */
