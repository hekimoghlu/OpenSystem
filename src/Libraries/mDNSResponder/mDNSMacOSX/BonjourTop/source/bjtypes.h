/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
//  bjtypes.h
//  TestTB
//
//  Created by Terrin Eager on 4/24/12.
//
//


#ifndef TestTB_bjtypes_h
#define TestTB_bjtypes_h
typedef bool BJ_BOOL;

typedef char BJ_INT8;
typedef unsigned char BJ_UINT8;

typedef short int BJ_INT16;
typedef unsigned short int BJ_UINT16;

typedef  int BJ_INT32;
typedef unsigned  int BJ_UINT32;

typedef  long long BJ_INT64;
typedef unsigned  long long BJ_UINT64;

#define DNS_NAME_OFFSET_MASK 0xc0

#define MAX_FRAME_SIZE 0x2800


enum BJ_COMPARE {BJ_GT,BJ_LT,BJ_EQUAL};

#define PF_GET_UINT8(pBuffer,offset) ( (BJ_UINT8)pBuffer[offset] )
#define PF_GET_UINT16(pBuffer,offset) ((((BJ_UINT16)pBuffer[offset]) << 8) | ((BJ_UINT16)pBuffer[offset+1]))
#define PF_GET_UINT32(pBuffer,offset) ((pBuffer[offset] << 24) | (pBuffer[offset+1] << 16) | (pBuffer[offset+2] << 8) | (pBuffer[offset+3]))


inline void endian_swap(BJ_UINT8& x)
{
    x = (x>>8) |(x<<8);
}

inline void endian_swap(BJ_UINT32& x)
{
    x = (x>>24) |
        ((x<<8) & 0x00FF0000) |
        ((x>>8) & 0x0000FF00) |
        (x<<24);
}


inline void endian_swap(BJ_UINT64& x)
{
    x = (x>>56) |
    ((x<<40) & 0x00FF000000000000) |
    ((x<<24) & 0x0000FF0000000000) |
    ((x<<8)  & 0x000000FF00000000) |
    ((x>>8)  & 0x00000000FF000000) |
    ((x>>24) & 0x0000000000FF0000) |
    ((x>>40) & 0x000000000000FF00) |
    (x<<56);
}

#endif
