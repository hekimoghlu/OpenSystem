/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
 * Copyright (c) 2004, International Business Machines Corporation
 * and others. All Rights Reserved.
 ********************************************************************/

/**
 * CollationServiceTest tests registration of collators.
 */

#ifndef _SVCCOLL
#define _SVCCOLL

#include "unicode/utypes.h"

#if !UCONFIG_NO_COLLATION

#include "intltest.h"

U_NAMESPACE_BEGIN

class StringEnumeration;

class CollationServiceTest: public IntlTest {
public:
    void runIndexedTest(int32_t index, UBool exec, const char* &name, char* /*par = nullptr */) override;

    void TestRegister();
    void TestRegisterFactory();
    void TestSeparateTree();

 private:
    int32_t checkStringEnumeration(const char* msg,
                                   StringEnumeration& iter,
                                   const char** expected,
                                   int32_t expectedCount);

    int32_t checkAvailable(const char* msg);
};

U_NAMESPACE_END

/* #if !UCONFIG_NO_COLLATION */
#endif

/* #ifndef _SVCCOLL */
#endif
