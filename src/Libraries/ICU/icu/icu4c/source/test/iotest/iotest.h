/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
 *   Copyright (C) 2004-2008, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 **********************************************************************
 *   file name:  iotest.h
 *   encoding:   UTF-8
 *   tab size:   8 (not used)
 *   indentation:4
 *
 *   created on: 2004apr06
 *   created by: George Rhoten
 */

#ifndef IOTEST_H
#define IOTEST_H 1

#include "unicode/utypes.h"
#include "unicode/ctest.h"

U_CFUNC void
addStringTest(TestNode** root);

U_CFUNC void
addFileTest(TestNode** root);

U_CFUNC void
addTranslitTest(TestNode** root);

U_CFUNC void
addStreamTests(TestNode** root);

U_CDECL_BEGIN
extern const UChar NEW_LINE[];
extern const char C_NEW_LINE[];
extern const char *STANDARD_TEST_FILE;
extern const char *MEDIUMNAME_TEST_FILE;
extern const char *LONGNAME_TEST_FILE;
U_CDECL_END

#define STANDARD_TEST_NUM_RANGE 1000


#endif
