/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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

#ifndef NEW_RESOURCEBUNDLETEST_H
#define NEW_RESOURCEBUNDLETEST_H

#include "intltest.h"

/**
 * Tests for class ResourceBundle
 **/
class NewResourceBundleTest: public IntlTest {
public:
    NewResourceBundleTest();
    virtual ~NewResourceBundleTest();
    
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    /** 
     * Perform several extensive tests using the subtest routine testTag
     **/
    void TestResourceBundles();
    /** 
     * Test construction of ResourceBundle accessing a custom test resource-file
     **/
    void TestConstruction();

    void TestIteration();

    void TestOtherAPI();

    void TestNewTypes();

    void TestGetByFallback();

    void TestFilter();

    void TestIntervalAliasFallbacks();

#if U_ENABLE_TRACING
    void TestTrace();
#endif

    void TestOpenDirectFillIn();
    void TestStackReuse();

private:
    /**
     * The assignment operator has no real implementation.
     * It is provided to make the compiler happy. Do not call.
     */
    NewResourceBundleTest& operator=(const NewResourceBundleTest&) { return *this; }

    /**
     * extensive subtests called by TestResourceBundles
     **/
    UBool testTag(const char* frag, UBool in_Root, UBool in_te, UBool in_te_IN);

    void record_pass();
    void record_fail();

    int32_t pass;
    int32_t fail;

};

#endif
