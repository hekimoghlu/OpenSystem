/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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


#ifndef TESTFORMATSMALLCLASSES_H
#define TESTFORMATSMALLCLASSES_H

#include "intltest.h"

/** 
 * tests 3 smaller classes in the format library
 **/
class TestFormatSmallClasses: public IntlTest {
    /**
     * runs tests in 4 local routines,
     * performs test for API and functionality of 3 smaller format classes:
     *    ParsePosition in test_ParsePosition(),
     *    FieldPosition in test_FieldPosition(),
     *    Formattable in test_Formattable().
     **/    
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;
};

#endif
