/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include <sys/types.h>

#if !defined(TRUE)
    #define TRUE	1
#endif

#if !defined(FALSE)
    #define FALSE	0
#endif

typedef u_int8_t	UInt8;
typedef u_int16_t	UniChar;
typedef u_int16_t	UInt16;
typedef u_int32_t	UInt32;
typedef u_int32_t	UniCharCount;
typedef unsigned char	Boolean;
typedef unsigned char	Str31[32];


#define kCFStringEncodingMacJapanese	1

/* Values for flags argument for the conversion functions below.  These can be combined, but the three NonSpacing behavior flags are exclusive.
*/
enum {
    kCFStringEncodingAllowLossyConversion = 1, // Uses fallback functions to substitutes non mappable chars
    kCFStringEncodingBasicDirectionLeftToRight = (1 << 1), // Converted with original direction left-to-right.
    kCFStringEncodingBasicDirectionRightToLeft = (1 << 2), // Converted with original direction right-to-left.
    kCFStringEncodingSubstituteCombinings = (1 << 3), // Uses fallback function to combining chars.
    kCFStringEncodingComposeCombinings = (1 << 4), // Checks mappable precomposed equivalents for decomposed sequences.  This is the default behavior.
    kCFStringEncodingIgnoreCombinings = (1 << 5), // Ignores combining chars.
    kCFStringEncodingUseCanonical = (1 << 6), // Always use canonical form
    kCFStringEncodingUseHFSPlusCanonical = (1 << 7), // Always use canonical form but leaves 0x2000 ranges
    kCFStringEncodingPrependBOM = (1 << 8), // Prepend BOM sequence (i.e. ISO2022KR)
    kCFStringEncodingDisableCorporateArea = (1 << 9), // Disable the usage of 0xF8xx area for Apple proprietary chars in converting to UniChar, resulting loosely mapping.
};

enum {
    kCFStringEncodingConversionSuccess = 0,
    kCFStringEncodingInvalidInputStream = 1,
    kCFStringEncodingInsufficientOutputBufferLength = 2,
    kCFStringEncodingConverterUnavailable = 3,
};


extern UInt32 __CFToMacJapanese(UInt32 flags, const UniChar *characters,
		UInt32 numChars, UInt8 *bytes, UInt32 maxByteLen, UInt32 *usedByteLen);

extern UInt32 __CFFromMacJapanese(UInt32 flags, const UInt8 *bytes, UInt32 numBytes,
		UniChar *characters, UInt32 maxCharLen, UInt32 *usedCharLen);

