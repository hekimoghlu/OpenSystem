/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
 * Copyright (c) 1997-2015, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _NUMBERFORMATROUNDTRIPTEST_
#define _NUMBERFORMATROUNDTRIPTEST_
 
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/numfmt.h"
#include "unicode/fmtable.h"
#include "intltest.h"

/** 
 * Performs round-trip tests for NumberFormat
 **/
class NumberFormatRoundTripTest : public IntlTest {    
    
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
public:

    static UBool verbose;
    static UBool STRING_COMPARE;
    static UBool EXACT_NUMERIC_COMPARE;
    static UBool DEBUG_VAR;
    static double MAX_ERROR;
    static double max_numeric_error;
    static double min_numeric_error;


    void start();

    void test(NumberFormat *fmt);
    void test(NumberFormat *fmt, double value);
    void test(NumberFormat *fmt, int32_t value);
    void test(NumberFormat *fmt, const Formattable& value);

    static double randomDouble(double range);
    static double proportionalError(const Formattable& a, const Formattable& b);
    static UnicodeString& typeOf(const Formattable& n, UnicodeString& result);
    static UnicodeString& escape(UnicodeString& s);

    static inline UBool
    isDouble(const Formattable& n)
    { return (n.getType() == Formattable::kDouble); }

    static inline UBool
    isLong(const Formattable& n)
    { return (n.getType() == Formattable::kLong); }

    /*
     * Return a random uint32_t
     **/
    static uint32_t randLong();

    /**
     * Return a random double 0 <= x < 1.0
     **/
    static double randFraction()
    {
        return static_cast<double>(randLong()) / static_cast<double>(0xFFFFFFFF);
    }

protected:
    UBool failure(UErrorCode status, const char* msg, UBool possibleDataError=false);

};

#endif /* #if !UCONFIG_NO_FORMATTING */
 
#endif // _NUMBERFORMATROUNDTRIPTEST_
//eof
