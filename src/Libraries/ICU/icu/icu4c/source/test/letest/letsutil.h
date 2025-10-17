/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
 *******************************************************************************
 *
 *   Copyright (C) 1999-2014, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *
 *******************************************************************************
 *   file name:  letsutil.h
 *
 *   created on: 04/25/2006
 *   created by: Eric R. Mader
 */

#ifndef __LETSUTIL_H
#define __LETSUTIL_H

#include "unicode/utypes.h"
#include "unicode/unistr.h"
#include "unicode/ubidi.h"

#include "layout/LETypes.h"
#include "layout/LEScripts.h"
#include "layout/LayoutEngine.h"
#include "layout/LELanguages.h"

#ifndef USING_ICULEHB
#include "OpenTypeLayoutEngine.h"
#endif

#include "letest.h"

char *getCString(const UnicodeString *uString);
char *getCString(const LEUnicode16 *uChars);
char *getUTF8String(const UnicodeString *uString);
void freeCString(char *cString);
le_bool getRTL(const UnicodeString &text);
le_int32 getLanguageCode(const char *lang);

#endif
