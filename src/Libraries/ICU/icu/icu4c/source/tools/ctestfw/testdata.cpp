/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
 * Copyright (c) 2002-2005, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/* Created by weiv 05/09/2002 */

#include "unicode/testdata.h"


TestData::TestData(const char* testName)
: name(testName),
fInfo(nullptr),
fCurrSettings(nullptr),
fCurrCase(nullptr),
fSettingsSize(0),
fCasesSize(0),
fCurrentSettings(0),
fCurrentCase(0)

{
}

TestData::~TestData() {
  delete fInfo;
  delete fCurrSettings;
  delete fCurrCase;
}

const char * TestData::getName() const
{
  return name;
}



RBTestData::RBTestData(const char* testName)
: TestData(testName),
fData(nullptr),
fHeaders(nullptr),
fSettings(nullptr),
fCases(nullptr)
{
}

RBTestData::RBTestData(UResourceBundle *data, UResourceBundle *headers, UErrorCode& status)
: TestData(ures_getKey(data)),
fData(data),
fHeaders(headers),
fSettings(nullptr),
fCases(nullptr)
{
  UErrorCode intStatus = U_ZERO_ERROR;
  UResourceBundle *currHeaders = ures_getByKey(data, "Headers", nullptr, &intStatus);
  if(intStatus == U_ZERO_ERROR) {
    ures_close(fHeaders);
    fHeaders = currHeaders;
  } else {
    intStatus = U_ZERO_ERROR;
  }
  fSettings = ures_getByKey(data, "Settings", nullptr, &intStatus);
  fSettingsSize = ures_getSize(fSettings);
  UResourceBundle *info = ures_getByKey(data, "Info", nullptr, &intStatus);
  if(U_SUCCESS(intStatus)) {
    fInfo = new RBDataMap(info, status);
  } else {
    intStatus = U_ZERO_ERROR;
  }
  fCases = ures_getByKey(data, "Cases", nullptr, &status);
  fCasesSize = ures_getSize(fCases);

  ures_close(info);
}


RBTestData::~RBTestData()
{
  ures_close(fData);
  ures_close(fHeaders);
  ures_close(fSettings);
  ures_close(fCases);
}

UBool RBTestData::getInfo(const DataMap *& info, UErrorCode &/*status*/) const
{
  if(fInfo) {
    info = fInfo;
    return true;
  } else {
    info = nullptr;
    return false;
  }
}

UBool RBTestData::nextSettings(const DataMap *& settings, UErrorCode &status)
{
  UErrorCode intStatus = U_ZERO_ERROR;
  UResourceBundle *data = ures_getByIndex(fSettings, fCurrentSettings++, nullptr, &intStatus);
  if(U_SUCCESS(intStatus)) {
    // reset the cases iterator
    fCurrentCase = 0;
    if(fCurrSettings == nullptr) {
      fCurrSettings = new RBDataMap(data, status);
    } else {
      ((RBDataMap *)fCurrSettings)->init(data, status);
    }
    ures_close(data);
    settings = fCurrSettings;
    return true;
  } else {
    settings = nullptr;
    return false;
  }
}

UBool RBTestData::nextCase(const DataMap *& nextCase, UErrorCode &status)
{
  UErrorCode intStatus = U_ZERO_ERROR;
  UResourceBundle *currCase = ures_getByIndex(fCases, fCurrentCase++, nullptr, &intStatus);
  if(U_SUCCESS(intStatus)) {
    if(fCurrCase == nullptr) {
      fCurrCase = new RBDataMap(fHeaders, currCase, status);
    } else {
      ((RBDataMap *)fCurrCase)->init(fHeaders, currCase, status);
    }
    ures_close(currCase);
    nextCase = fCurrCase;
    return true;
  } else {
    nextCase = nullptr;
    return false;
  }
}


