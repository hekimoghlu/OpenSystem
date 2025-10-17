/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#ifndef _UAPI__ASM_SVE_CONTEXT_H
#define _UAPI__ASM_SVE_CONTEXT_H
#include <linux/types.h>
#define __SVE_VQ_BYTES 16
#define __SVE_VQ_MIN 1
#define __SVE_VQ_MAX 512
#define __SVE_VL_MIN (__SVE_VQ_MIN * __SVE_VQ_BYTES)
#define __SVE_VL_MAX (__SVE_VQ_MAX * __SVE_VQ_BYTES)
#define __SVE_NUM_ZREGS 32
#define __SVE_NUM_PREGS 16
#define __sve_vl_valid(vl) ((vl) % __SVE_VQ_BYTES == 0 && (vl) >= __SVE_VL_MIN && (vl) <= __SVE_VL_MAX)
#define __sve_vq_from_vl(vl) ((vl) / __SVE_VQ_BYTES)
#define __sve_vl_from_vq(vq) ((vq) * __SVE_VQ_BYTES)
#define __SVE_ZREG_SIZE(vq) ((__u32) (vq) * __SVE_VQ_BYTES)
#define __SVE_PREG_SIZE(vq) ((__u32) (vq) * (__SVE_VQ_BYTES / 8))
#define __SVE_FFR_SIZE(vq) __SVE_PREG_SIZE(vq)
#define __SVE_ZREGS_OFFSET 0
#define __SVE_ZREG_OFFSET(vq,n) (__SVE_ZREGS_OFFSET + __SVE_ZREG_SIZE(vq) * (n))
#define __SVE_ZREGS_SIZE(vq) (__SVE_ZREG_OFFSET(vq, __SVE_NUM_ZREGS) - __SVE_ZREGS_OFFSET)
#define __SVE_PREGS_OFFSET(vq) (__SVE_ZREGS_OFFSET + __SVE_ZREGS_SIZE(vq))
#define __SVE_PREG_OFFSET(vq,n) (__SVE_PREGS_OFFSET(vq) + __SVE_PREG_SIZE(vq) * (n))
#define __SVE_PREGS_SIZE(vq) (__SVE_PREG_OFFSET(vq, __SVE_NUM_PREGS) - __SVE_PREGS_OFFSET(vq))
#define __SVE_FFR_OFFSET(vq) (__SVE_PREGS_OFFSET(vq) + __SVE_PREGS_SIZE(vq))
#endif
