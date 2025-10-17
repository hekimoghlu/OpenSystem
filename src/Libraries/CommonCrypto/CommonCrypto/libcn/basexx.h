/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#ifndef CommonNumerics_basexx_h
#define CommonNumerics_basexx_h

#include <stdint.h>
#include <stddef.h>
#include "../lib/ccDispatch.h"
#include <CommonNumerics/CommonNumerics.h>
#include <CommonNumerics/CommonBaseXX.h>

typedef struct encoderConstants_t {
    uint32_t    baseNum;
    uint32_t    log;
    uint32_t    inputBlocksize;
    uint32_t    outputBlocksize;
    uint8_t     basemask;
} encoderConstants;

typedef struct baseEncoder_t {
    const char *name;
    CNEncodings encoding;
    const char *charMap;
    const encoderConstants *values;
    uint8_t baseNum;
    uint8_t padding;
} BaseEncoder;

typedef BaseEncoder *BaseEncoderRefCustom;
typedef const BaseEncoder *BaseEncoderRef;

#define CC_BASE_REVERSE_MAP_SIZE 256
// This manages a global context for encoders.
typedef struct coderFrame_t {
    uint8_t reverseMap[CC_BASE_REVERSE_MAP_SIZE];
    BaseEncoderRef encoderRef;
} BaseEncoderFrame, *CoderFrame;

extern const BaseEncoder defaultBase64;
extern const BaseEncoder defaultBase32; // RFC 4678 Base32Alphabet
extern const BaseEncoder recoveryBase32;
extern const BaseEncoder hexBase32;
extern const BaseEncoder defaultBase16;
void setReverseMap(CoderFrame frame);


#endif
