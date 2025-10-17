/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
/*
 **********************************************************************
 *   Copyright (C) 2005-2012, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 **********************************************************************
 */

#ifndef __CSDETEST_H
#define __CSDETEST_H

#include "unicode/utypes.h"
#include "unicode/unistr.h"

#include "intltest.h"

class CharsetDetectionTest: public IntlTest {
public:
  
    CharsetDetectionTest();
    virtual ~CharsetDetectionTest();

    virtual void runIndexedTest(int32_t index, UBool exec, const char* &name, char* par = nullptr ) override;

    virtual void ConstructionTest();
    virtual void UTF8Test();
    virtual void UTF16Test();
    virtual void C1BytesTest();
    virtual void InputFilterTest();
    virtual void DetectionTest();
    virtual void IBM424Test();
    virtual void IBM420Test();
    virtual void Ticket6394Test();
    virtual void Ticket6954Test();
    virtual void Ticket21823Test();

private:
    void checkEncoding(const UnicodeString &testString,
                       const UnicodeString &encoding, const UnicodeString &id);

    virtual const char *getPath(char buffer[2048], const char *filename);

};

#endif
