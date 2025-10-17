/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/********************************************************************
 * COPYRIGHT: 
 * Copyright (c) 1997-2001, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
/********************************************************************************
*
* File NCCBTST.H
*
* Modification History:
*        Name                     Description            
*     Madhu Katragadda           creation
*********************************************************************************
*/
#ifndef _NCNVFBTS
#define _NCNVFBTS
/* C API TEST FOR FALL BACK ROUTINES OF CODESET CONVERSION COMPONENT */
#include "cintltst.h"
#include "unicode/utypes.h"

static void TestConverterFallBack(void);
static void TestConvertFallBackWithBufferSizes(int32_t outsize, int32_t insize );
static UBool testConvertFromUnicode(const UChar *source, int sourceLen,  const uint8_t *expect, int expectLen, 
                const char *codepage, UBool fallback, const int32_t *expectOffsets);
static UBool testConvertToUnicode( const uint8_t *source, int sourcelen, const UChar *expect, int expectlen, 
               const char *codepage, UBool fallback, const int32_t *expectOffsets);


static void printSeq(const unsigned char* a, int len);
static void printUSeq(const UChar* a, int len);
static void printSeqErr(const unsigned char* a, int len);
static void printUSeqErr(const UChar* a, int len);
static void setNuConvTestName(const char *codepage, const char *direction);


#endif
