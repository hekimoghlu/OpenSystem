/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
 * Copyright (c) 1997-2010, International Business Machines Corporation and
 * others. All Rights Reserved.
 * Copyright (C) 2010 , Yahoo! Inc. 
 ********************************************************************/

#ifndef _SELECTFORMATTEST
#define _SELECTFORMATTEST

#include "unicode/utypes.h"
#include "unicode/selfmt.h"


#if !UCONFIG_NO_FORMATTING

#include "intltest.h"

/**
 * Test basic functionality of various API functions
 **/
class SelectFormatTest : public IntlTest {
    void runIndexedTest( int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

private:
    /**
     * Performs tests on many API functions, see detailed comments in source code
     **/
    void selectFormatAPITest(/* char* par */);
    void selectFormatUnitTest(/* char* par */);
};

#endif /* #if !UCONFIG_NO_FORMATTING */

#endif
