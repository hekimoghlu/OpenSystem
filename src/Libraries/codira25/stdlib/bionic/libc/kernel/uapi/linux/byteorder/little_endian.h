/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#ifndef _UAPI_LINUX_BYTEORDER_LITTLE_ENDIAN_H
#define _UAPI_LINUX_BYTEORDER_LITTLE_ENDIAN_H
#ifndef __LITTLE_ENDIAN
#define __LITTLE_ENDIAN 1234
#endif
#ifndef __LITTLE_ENDIAN_BITFIELD
#define __LITTLE_ENDIAN_BITFIELD
#endif
#include <linux/stddef.h>
#include <linux/types.h>
#include <linux/swab.h>
#define __constant_htonl(x) (( __be32) ___constant_swab32((x)))
#define __constant_ntohl(x) ___constant_swab32(( __be32) (x))
#define __constant_htons(x) (( __be16) ___constant_swab16((x)))
#define __constant_ntohs(x) ___constant_swab16(( __be16) (x))
#define __constant_cpu_to_le64(x) (( __le64) (__u64) (x))
#define __constant_le64_to_cpu(x) (( __u64) (__le64) (x))
#define __constant_cpu_to_le32(x) (( __le32) (__u32) (x))
#define __constant_le32_to_cpu(x) (( __u32) (__le32) (x))
#define __constant_cpu_to_le16(x) (( __le16) (__u16) (x))
#define __constant_le16_to_cpu(x) (( __u16) (__le16) (x))
#define __constant_cpu_to_be64(x) (( __be64) ___constant_swab64((x)))
#define __constant_be64_to_cpu(x) ___constant_swab64(( __u64) (__be64) (x))
#define __constant_cpu_to_be32(x) (( __be32) ___constant_swab32((x)))
#define __constant_be32_to_cpu(x) ___constant_swab32(( __u32) (__be32) (x))
#define __constant_cpu_to_be16(x) (( __be16) ___constant_swab16((x)))
#define __constant_be16_to_cpu(x) ___constant_swab16(( __u16) (__be16) (x))
#define __cpu_to_le64(x) (( __le64) (__u64) (x))
#define __le64_to_cpu(x) (( __u64) (__le64) (x))
#define __cpu_to_le32(x) (( __le32) (__u32) (x))
#define __le32_to_cpu(x) (( __u32) (__le32) (x))
#define __cpu_to_le16(x) (( __le16) (__u16) (x))
#define __le16_to_cpu(x) (( __u16) (__le16) (x))
#define __cpu_to_be64(x) (( __be64) __swab64((x)))
#define __be64_to_cpu(x) __swab64(( __u64) (__be64) (x))
#define __cpu_to_be32(x) (( __be32) __swab32((x)))
#define __be32_to_cpu(x) __swab32(( __u32) (__be32) (x))
#define __cpu_to_be16(x) (( __be16) __swab16((x)))
#define __be16_to_cpu(x) __swab16(( __u16) (__be16) (x))
#define __cpu_to_le64s(x) do { (void) (x); } while(0)
#define __le64_to_cpus(x) do { (void) (x); } while(0)
#define __cpu_to_le32s(x) do { (void) (x); } while(0)
#define __le32_to_cpus(x) do { (void) (x); } while(0)
#define __cpu_to_le16s(x) do { (void) (x); } while(0)
#define __le16_to_cpus(x) do { (void) (x); } while(0)
#define __cpu_to_be64s(x) __swab64s((x))
#define __be64_to_cpus(x) __swab64s((x))
#define __cpu_to_be32s(x) __swab32s((x))
#define __be32_to_cpus(x) __swab32s((x))
#define __cpu_to_be16s(x) __swab16s((x))
#define __be16_to_cpus(x) __swab16s((x))
#endif
