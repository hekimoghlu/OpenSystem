/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
 * Copyright (c) 2007, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/**
 * DataDrivenFormatTest is a test class that uses data stored in resource
 * bundles to perform testing. For more details on data structure, see
 * source/test/testdata/calendar.txt
 */

#ifndef _INTLTESTDATADRIVENFORMAT
#define _INTLTESTDATADRIVENFORMAT

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "tsdate.h"
#include "uvector.h"
#include "unicode/format.h"
//#include "fldset.h"

class TestDataModule;
class TestData;
class DataMap;
//class DateTimeStyle;

class DataDrivenFormatTest : public IntlTest {
    void runIndexedTest(int32_t index, UBool exec, const char* &name,
            char* par = nullptr) override;
public:
    DataDrivenFormatTest();
    virtual ~DataDrivenFormatTest();
protected:

    void DataDrivenTest(char *par);
    void processTest(TestData *testData);
private:
    void testConvertDate(TestData *testData, const DataMap *settings, UBool fmt);
//    void testOps(TestData *testData, const DataMap *settings);
//    void testConvert(int32_t n, const FormatFieldsSet &fromSet,
//            Format *fromCal, const FormatFieldsSet &toSet, Format *toCal,
//            UBool fwd);
private:
    TestDataModule *driver;
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
