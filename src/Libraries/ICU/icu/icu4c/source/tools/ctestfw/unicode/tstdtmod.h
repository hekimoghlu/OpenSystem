/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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

/* Base class for data driven tests */

#ifndef U_TESTFW_TESTMODULE
#define U_TESTFW_TESTMODULE

#include "unicode/unistr.h"
#include "unicode/ures.h"
#include "unicode/testtype.h"
#include "unicode/testdata.h"
#include "unicode/datamap.h"
#include "unicode/testlog.h"


/* This class abstracts the actual organization of the  
 * data for data driven tests                           
 */


class DataMap;
class TestData;


/** Main data driven test class. Corresponds to one named data 
 *  unit (such as a resource bundle. It is instantiated using  
 *  a factory method getTestDataModule 
 */
class T_CTEST_EXPORT_API TestDataModule {
  const char* testName;

protected:
  DataMap *fInfo;
  TestLog& fLog;

public:
  /** Factory method. 
   *  @param name name of the test module. Usually name of a resource bundle or a XML file 
   *  @param log a logging class, used for internal error reporting.                       
   *  @param status if something goes wrong, status will be set                            
   *  @return a TestDataModule object. Use it to get test data from it                     
   */
  static TestDataModule *getTestDataModule(const char* name, TestLog& log, UErrorCode &status);
  virtual ~TestDataModule();

protected:
  TestDataModule(const char* name, TestLog& log, UErrorCode& status);

public:
  /** Name of this TestData module. 
   *  @return a name 
   */
  const char * getName() const;

  /** Get a pointer to an object owned DataMap that contains more information on this module 
   *  Usual fields are "Description", "LongDescription", "Settings". Also, if containing a   
   *  field "Headers" these will be used as the default headers, so that you don't have to   
   *  to specify per test headers.                                                           
   *  @param info pass in a const DataMap pointer. If no info, it will be set to nullptr
   */
  virtual UBool getInfo(const DataMap *& info, UErrorCode &status) const = 0;

  /** Create a test data object from an index. Helpful for integrating tests with current 
   *  intltest framework which addresses the tests by index.                              
   *  @param index index of the test to be instantiated                                   
   *  @return an instantiated TestData object, ready to provide settings and cases for    
   *          the tests.                                                                  
   */
  virtual TestData* createTestData(int32_t index, UErrorCode &status) const = 0;

  /** Create a test data object from a name.                              
   *  @param name name of the test to be instantiated                                     
   *  @return an instantiated TestData object, ready to provide settings and cases for    
   *          the tests.                                                                  
   */
  virtual TestData* createTestData(const char* name, UErrorCode &status) const = 0;
};

class T_CTEST_EXPORT_API RBTestDataModule : public TestDataModule {
public:
  virtual ~RBTestDataModule();

public:
  RBTestDataModule(const char* name, TestLog& log, UErrorCode& status);

public:
  virtual UBool getInfo(const DataMap *& info, UErrorCode &status) const override;

  virtual TestData* createTestData(int32_t index, UErrorCode &status) const override;
  virtual TestData* createTestData(const char* name, UErrorCode &status) const override;

private:
  UResourceBundle *getTestBundle(const char* bundleName, UErrorCode &status);

private:
  UResourceBundle *fModuleBundle;
  UResourceBundle *fTestData;
  UResourceBundle *fInfoRB;
  UBool fDataTestValid;
  char *tdpath;

/* const char* fTestName;*/ /* See name */
  int32_t fNumberOfTests;

};


#endif

