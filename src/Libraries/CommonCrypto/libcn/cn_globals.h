/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#ifndef CommonNumerics_cn_globals_h
#define CommonNumerics_cn_globals_h


/*
 *  cn_globals.h - CommonNumerics global DATA
 */

#include <asl.h>
#include "crc.h"
#include "basexx.h"
#include <CommonNumerics/CommonCRC.h>

#if __has_include(<os/alloc_once_private.h>)
#include <os/alloc_once_private.h>
#if defined(OS_ALLOC_ONCE_KEY_LIBCOMMONNUMERICS)
#define _LIBCOMMONNUMERICS_HAS_ALLOC_ONCE 1
#endif
#endif

#define CN_SUPPORTED_CRCS kCN_CRC_64_ECMA_182+1
#define CN_STANDARD_BASE_ENCODERS kCNEncodingBase16+1

struct cn_globals_s {
	// CommonCRC.c
    dispatch_once_t crc_init;
    crcInfo crcSelectionTab[CN_SUPPORTED_CRCS];
    dispatch_once_t basexx_init;
    BaseEncoderFrame encoderTab[CN_STANDARD_BASE_ENCODERS];
};
typedef struct cn_globals_s *cn_globals_t;

__attribute__((__pure__))
static inline cn_globals_t
_cn_globals(void) {
#if _LIBCOMMONNUMERICS_HAS_ALLOC_ONCE
	return (cn_globals_t) os_alloc_once(OS_ALLOC_ONCE_KEY_LIBCOMMONNUMERICS,
                                        sizeof(struct cn_globals_s),
                                        NULL);
#else
	static struct cn_globals_s storage;
	return &storage;
#endif
}

#endif
