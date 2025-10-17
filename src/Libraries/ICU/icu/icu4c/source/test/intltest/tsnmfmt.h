/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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
 * Copyright (c) 1997-2003, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

#ifndef _INTLTESTNUMBERFORMAT
#define _INTLTESTNUMBERFORMAT


#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/numfmt.h"
#include "unicode/locid.h"
#include "intltest.h"


/**
 * This test does round-trip testing (format -> parse -> format -> parse -> etc.) of
 * NumberFormat.
 */
class IntlTestNumberFormat: public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:

    /**
     *  call tryIt with many variations, called by testLocale
     **/
    void testFormat(/* char* par */);
    /**
     *  perform tests using aNumber and fFormat, called in many variations
     **/
    void tryIt(double aNumber);
    /**
     *  perform tests using aNumber and fFormat, called in many variations
     **/
    void tryIt(int32_t aNumber);
    /**
     *  test NumberFormat::getAvailableLocales
     **/
    void testAvailableLocales(/* char* par */);
    /**
     *  call testLocale for all locales
     **/
    void monsterTest(/* char *par */);
    /**
     *  call testFormat for currency, percent and plain number instances
     **/
    void testLocale(/* char *par, */const Locale& locale, const UnicodeString& localeName);

    NumberFormat*   fFormat;
    UErrorCode      fStatus;
    Locale          fLocale;

public:

    virtual ~IntlTestNumberFormat();

    /*
     * Return a random double that isn't too large.
     */
    static double getSafeDouble(double smallerThanMax);

    /*
     * Return a random double
     **/
    static double randDouble();

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

};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
