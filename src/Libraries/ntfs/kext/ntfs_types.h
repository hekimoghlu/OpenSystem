/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#ifndef _OSX_NTFS_TYPES_H
#define _OSX_NTFS_TYPES_H

#include <sys/types.h>

#include <mach/boolean.h>

/* Define our fixed size types. */
typedef u_int8_t u8;
typedef u_int16_t u16;
typedef u_int32_t u32;
typedef u_int64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

/*
 * Define our fixed size, little-endian types.  Note we define the signed types
 * to be unsigned so we do not get sign extension on endianness conversions.
 * We do not bother with eight-bit, little-endian types as endianness does not
 * apply for eight-bit types.
 */
typedef u_int16_t le16;
typedef u_int32_t le32;
typedef u_int64_t le64;
typedef u_int16_t sle16;
typedef u_int32_t sle32;
typedef u_int64_t sle64;

/*
 * Define our fixed size, big-endian types.  Note we define the signed types to
 * be unsigned so we do not get sign extension on endianness conversions.  We
 * do not bother with eight-bit, big-endian types as endianness does not apply
 * for eight-bit types.
 */
typedef u_int16_t be16;
typedef u_int32_t be32;
typedef u_int64_t be64;
typedef u_int16_t sbe16;
typedef u_int32_t sbe32;
typedef u_int64_t sbe64;

/* 2-byte Unicode character type. */
typedef le16 ntfschar;
#define NTFSCHAR_SIZE_SHIFT 1

/*
 * Clusters are signed 64-bit values on NTFS volumes.  We define two types, LCN
 * and VCN, to allow for type checking and better code readability.
 */
typedef s64 VCN;
typedef sle64 leVCN;
typedef s64 LCN;
typedef sle64 leLCN;

/*
 * The NTFS journal $LogFile uses log sequence numbers which are signed 64-bit
 * values.  We define our own type LSN, to allow for type checking and better
 * code readability.
 */
typedef s64 LSN;
typedef sle64 leLSN;

/*
 * The NTFS transaction log $UsnJrnl uses usns which are signed 64-bit values.
 * We define our own type USN, to allow for type checking and better code
 * readability.
 */
typedef s64 USN;
typedef sle64 leUSN;

/* Our boolean type. */
typedef boolean_t BOOL;

#endif /* !_OSX_NTFS_TYPES_H */
