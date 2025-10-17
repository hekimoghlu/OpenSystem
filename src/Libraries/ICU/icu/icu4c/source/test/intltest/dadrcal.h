/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
 * DataDrivenCalendarTest is a test class that uses data stored in resource
 * bundles to perform testing. For more details on data structure, see
 * source/test/testdata/calendar.txt
 */

#ifndef _INTLTESTDATADRIVENCALENDAR
#define _INTLTESTDATADRIVENCALENDAR

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "tsdate.h"
#include "uvector.h"
#include "unicode/calendar.h"
#include "fldset.h"

class TestDataModule;
class TestData;
class DataMap;
class CalendarFieldsSet;

class DataDrivenCalendarTest : public IntlTest {
	void runIndexedTest(int32_t index, UBool exec, const char* &name,
			char* par = nullptr) override;
public:
	DataDrivenCalendarTest();
	virtual ~DataDrivenCalendarTest();
protected:

	void DataDrivenTest(char *par);
	void processTest(TestData *testData);
private:
	void testConvert(TestData *testData, const DataMap *settings, UBool fwd);
	void testOps(TestData *testData, const DataMap *settings);
	void testConvert(int32_t n, const CalendarFieldsSet &fromSet,
			Calendar *fromCal, const CalendarFieldsSet &toSet, Calendar *toCal,
			UBool fwd);
private:
	TestDataModule *driver;
};

#endif /* #if !UCONFIG_NO_COLLATION */

#endif
