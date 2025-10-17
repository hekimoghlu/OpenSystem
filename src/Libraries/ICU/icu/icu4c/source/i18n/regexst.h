/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
//
//  regexst.h
//
//  Copyright (C) 2003-2010, International Business Machines Corporation and others.
//  All Rights Reserved.
//
//  This file contains declarations for the class RegexStaticSets
//
//  This class is internal to the regular expression implementation.
//  For the public Regular Expression API, see the file "unicode/regex.h"
//
//  RegexStaticSets groups together the common UnicodeSets that are needed
//   for compiling or executing RegularExpressions.  This grouping simplifies
//   the thread safe lazy creation and sharing of these sets across
//   all instances of regular expressions.
//

#ifndef REGEXST_H
#define REGEXST_H

#include "unicode/utypes.h"
#include "unicode/utext.h"
#if !UCONFIG_NO_REGULAR_EXPRESSIONS

#include "regeximp.h"
#include "regexcst.h"

U_NAMESPACE_BEGIN

class  UnicodeSet;


class RegexStaticSets : public UMemory {
public:
    static RegexStaticSets *gStaticSets;  // Ptr to all lazily initialized constant
                                          //   shared sets.

    RegexStaticSets(UErrorCode *status);         
    ~RegexStaticSets();
    static void    initGlobals(UErrorCode *status);

    UnicodeSet    fPropSets[URX_LAST_SET] {};      // The sets for common regex items, e.g. \s
    Regex8BitSet  fPropSets8[URX_LAST_SET] {};     // Fast bitmap sets for latin-1 range for above.

    UnicodeSet    fRuleSets[kRuleSet_count] {};    // Sets used while parsing regexp patterns.
    UnicodeSet    fUnescapeCharSet {};             // Set of chars handled by unescape when
                                                   //   encountered with a \ in a pattern.
    UnicodeSet    *fRuleDigitsAlias {};
    UText         *fEmptyText {};                  // An empty string, to be used when a matcher
                                                   //   is created with no input.

};


U_NAMESPACE_END
#endif   // !UCONFIG_NO_REGULAR_EXPRESSIONS
#endif   // REGEXST_H

