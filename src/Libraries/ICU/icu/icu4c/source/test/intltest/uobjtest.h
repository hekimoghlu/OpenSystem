/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
 * Copyright (c) 2002-2010, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/


#ifndef _UOBJECTTEST_
#define _UOBJECTTEST_

#include "intltest.h"

/** 
 * Test uobjtest.h
 **/
class UObjectTest : public IntlTest {
    // IntlTest override
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par ) override;
private:
    // tests
    void testIDs();
    void testUMemory();
    void TestMFCCompatibility();
    void TestCompilerRTTI();

    //helper

    /**
     * @param obj The UObject to be tested
     * @param className The name of the class being tested 
     * @param factory String version of obj, for exanple   "new UFoo(1,3,4)". nullptr if object is abstract.
     * @param staticID The result of class :: getStaticClassID
     * @return Returns obj, suitable for deletion
     */
    UObject *testClass(UObject *obj,
               const char *className, const char *factory, 
               UClassID staticID);

    UObject *testClassNoClassID(UObject *obj,
               const char *className, const char *factory);
};

#endif
//eof
