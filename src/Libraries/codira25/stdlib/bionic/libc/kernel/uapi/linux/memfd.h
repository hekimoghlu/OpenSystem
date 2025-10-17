/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#ifndef _UAPI_LINUX_MEMFD_H
#define _UAPI_LINUX_MEMFD_H
#include <asm-generic/hugetlb_encode.h>
#define MFD_CLOEXEC 0x0001U
#define MFD_ALLOW_SEALING 0x0002U
#define MFD_HUGETLB 0x0004U
#define MFD_NOEXEC_SEAL 0x0008U
#define MFD_EXEC 0x0010U
#define MFD_HUGE_SHIFT HUGETLB_FLAG_ENCODE_SHIFT
#define MFD_HUGE_MASK HUGETLB_FLAG_ENCODE_MASK
#define MFD_HUGE_64KB HUGETLB_FLAG_ENCODE_64KB
#define MFD_HUGE_512KB HUGETLB_FLAG_ENCODE_512KB
#define MFD_HUGE_1MB HUGETLB_FLAG_ENCODE_1MB
#define MFD_HUGE_2MB HUGETLB_FLAG_ENCODE_2MB
#define MFD_HUGE_8MB HUGETLB_FLAG_ENCODE_8MB
#define MFD_HUGE_16MB HUGETLB_FLAG_ENCODE_16MB
#define MFD_HUGE_32MB HUGETLB_FLAG_ENCODE_32MB
#define MFD_HUGE_256MB HUGETLB_FLAG_ENCODE_256MB
#define MFD_HUGE_512MB HUGETLB_FLAG_ENCODE_512MB
#define MFD_HUGE_1GB HUGETLB_FLAG_ENCODE_1GB
#define MFD_HUGE_2GB HUGETLB_FLAG_ENCODE_2GB
#define MFD_HUGE_16GB HUGETLB_FLAG_ENCODE_16GB
#endif
