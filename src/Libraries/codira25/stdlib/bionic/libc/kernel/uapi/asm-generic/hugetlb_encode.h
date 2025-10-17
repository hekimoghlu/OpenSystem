/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#ifndef _ASM_GENERIC_HUGETLB_ENCODE_H_
#define _ASM_GENERIC_HUGETLB_ENCODE_H_
#define HUGETLB_FLAG_ENCODE_SHIFT 26
#define HUGETLB_FLAG_ENCODE_MASK 0x3f
#define HUGETLB_FLAG_ENCODE_16KB (14U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_64KB (16U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_512KB (19U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_1MB (20U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_2MB (21U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_8MB (23U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_16MB (24U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_32MB (25U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_256MB (28U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_512MB (29U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_1GB (30U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_2GB (31U << HUGETLB_FLAG_ENCODE_SHIFT)
#define HUGETLB_FLAG_ENCODE_16GB (34U << HUGETLB_FLAG_ENCODE_SHIFT)
#endif
