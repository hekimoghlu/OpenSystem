/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#ifndef _NCCBTST
#define _NCCBTST

#include "unicode/utypes.h"
#include "unicode/ucnv.h"

/* C API TEST FOR CALL BACK ROUTINES OF CODESET CONVERSION COMPONENT */
#include "cintltst.h"


static void TestSkipCallBack(void);
static void TestStopCallBack(void);
static void TestSubCallBack(void);
static void TestSubWithValueCallBack(void);
static void TestLegalAndOtherCallBack(void);
static void TestSingleByteCallBack(void);


static void TestSkip(int32_t inputsize, int32_t outputsize);

static void TestStop(int32_t inputsize, int32_t outputsize);

static void TestSub(int32_t inputsize, int32_t outputsize);

static void TestSubWithValue(int32_t inputsize, int32_t outputsize);

static void TestLegalAndOthers(int32_t inputsize, int32_t outputsize);
static void TestSingleByte(int32_t inputsize, int32_t outputsize);
static void TestEBCDIC_STATEFUL_Sub(int32_t inputsize, int32_t outputsize);

/* Following will return false *only* on a mismatch. They will return true on any other error OR success, because
 * the error would have been emitted to log_err separately. */

UBool testConvertFromUnicode(const UChar *source, int sourceLen,  const uint8_t *expect, int expectLen, 
                const char *codepage, UConverterFromUCallback callback, const int32_t *expectOffsets,
                const char *mySubChar, int8_t len);


UBool testConvertToUnicode( const uint8_t *source, int sourcelen, const UChar *expect, int expectlen, 
               const char *codepage, UConverterToUCallback callback, const int32_t *expectOffsets,
               const char *mySubChar, int8_t len);

UBool testConvertFromUnicodeWithContext(const UChar *source, int sourceLen,  const uint8_t *expect, int expectLen, 
                const char *codepage, UConverterFromUCallback callback , const int32_t *expectOffsets, 
                const char *mySubChar, int8_t len, const void* context, UErrorCode expectedError);

UBool testConvertToUnicodeWithContext( const uint8_t *source, int sourcelen, const UChar *expect, int expectlen, 
               const char *codepage, UConverterToUCallback callback, const int32_t *expectOffsets,
               const char *mySubChar, int8_t len, const void* context, UErrorCode expectedError);

static void printSeq(const uint8_t* a, int len);
static void printUSeq(const UChar* a, int len);
static void printSeqErr(const uint8_t* a, int len);
static void printUSeqErr(const UChar* a, int len);
static void setNuConvTestName(const char *codepage, const char *direction);


#endif
