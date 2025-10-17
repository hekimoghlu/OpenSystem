/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
 * Copyright (c) 2002-2006, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/

/* Created by weiv 05/09/2002 */

/* Base class for data driven tests */

#ifndef U_TESTFW_TESTDATA
#define U_TESTFW_TESTDATA

#include "unicode/tstdtmod.h"
#include "unicode/datamap.h"


 /** This is the class that abstracts one of the tests in a data file 
  *  It is usually instantiated using TestDataModule::CreateTestData method 
  *  This class provides two important methods: nextSettings and nextCase 
  *  Usually, one walks through all settings and executes all cases for 
  *  each setting. Each call to nextSettings resets the cases iterator.
  *  Individual test cases have to have the same number of fields as the
  *  number of entries in headers. Default headers can be specified in 
  *  the TestDataModule info section. The default headers will be overridden
  *  by per-test headers. 
  *  Example:                                             
  *  DataMap *settings = nullptr;
  *  DataMap *cases = nullptr;
  *  while(nextSettings(settings, status)) {              
  *    // set settings for the subtest                    
  *    while(nextCase(cases, status) {                    
  *      // process testcase                              
  *    }                                                  
  *   }                                                   
  */

class T_CTEST_EXPORT_API TestData {
  const char* name;

protected:
  DataMap *fInfo;
  DataMap *fCurrSettings;
  DataMap *fCurrCase;
  int32_t fSettingsSize;
  int32_t fCasesSize;
  int32_t fCurrentSettings;
  int32_t fCurrentCase;
  /** constructor - don't use */
  TestData(const char* name);

public:
  virtual ~TestData();

  const char* getName() const;

  /** Get a pointer to an object owned DataMap that contains more information on this
   *  TestData object.
   *  Usual fields is "Description".                                   
   *  @param info pass in a const DataMap pointer. If no info, it will be set to nullptr
   */
  virtual UBool getInfo(const DataMap *& info, UErrorCode &status) const = 0;

  /** Gets the next set of settings for the test. Resets the cases iterator. 
   *  DataMap is owned by the object and should not be deleted. 
   *  @param settings a DataMap pointer provided by the user. Will be nullptr if
   *                  no more settings are available.
   *  @param status for reporting unexpected errors.
   *  @return A boolean, true if there are settings, false if there is no more 
   *          settings. 
   */
  virtual UBool nextSettings(const DataMap *& settings, UErrorCode &status) = 0;

  /** Gets the next test case. 
   *  DataMap is owned by the object and should not be deleted. 
   *  @param data a DataMap pointer provided by the user. Will be nullptr if 
   *                  no more cases are available.
   *  @param status for reporting unexpected errors.
   *  @return A boolean, true if there are cases, false if there is no more 
   *          cases. 
   */
  virtual UBool nextCase(const DataMap *& data, UErrorCode &status) = 0;
};

// implementation of TestData that uses resource bundles

class T_CTEST_EXPORT_API RBTestData : public TestData {
  UResourceBundle *fData;
  UResourceBundle *fHeaders;
  UResourceBundle *fSettings;
  UResourceBundle *fCases;

public:
  RBTestData(const char* name);
  RBTestData(UResourceBundle *data, UResourceBundle *headers, UErrorCode& status);
private:
//  RBTestData() {};
//  RBTestData(const RBTestData& original) {};
  RBTestData& operator=(const RBTestData& /*original*/);

public:
  virtual ~RBTestData();

  virtual UBool getInfo(const DataMap *& info, UErrorCode &status) const override;

  virtual UBool nextSettings(const DataMap *& settings, UErrorCode &status) override;
  virtual UBool nextCase(const DataMap *& nextCase, UErrorCode &status) override;
};

#endif

