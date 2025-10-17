/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
 *   Copyright (C) 2005-2015, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 **********************************************************************
 */

#ifndef __CSR2022_H
#define __CSR2022_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_CONVERSION

#include "csrecog.h"

U_NAMESPACE_BEGIN

class CharsetMatch;

/**
 *  class CharsetRecog_2022  part of the ICU charset detection implementation.
 *                           This is a superclass for the individual detectors for
 *                           each of the detectable members of the ISO 2022 family
 *                           of encodings.
 * 
 *                           The separate classes are nested within this class.
 * 
 * @internal
 */
class CharsetRecog_2022 : public CharsetRecognizer
{

public:    
    virtual ~CharsetRecog_2022() = 0;

protected:

    /**
     * Matching function shared among the 2022 detectors JP, CN and KR
     * Counts up the number of legal an unrecognized escape sequences in
     * the sample of text, and computes a score based on the total number &
     * the proportion that fit the encoding.
     * 
     * 
     * @param text the byte buffer containing text to analyse
     * @param textLen  the size of the text in the byte.
     * @param escapeSequences the byte escape sequences to test for.
     * @return match quality, in the range of 0-100.
     */
    int32_t match_2022(const uint8_t *text,
                       int32_t textLen,
                       const uint8_t escapeSequences[][5],
                       int32_t escapeSequences_length) const;

};

class CharsetRecog_2022JP :public CharsetRecog_2022
{
public:
    virtual ~CharsetRecog_2022JP();

    const char *getName() const override;

    UBool match(InputText *textIn, CharsetMatch *results) const override;
};

#if !UCONFIG_ONLY_HTML_CONVERSION
class CharsetRecog_2022KR :public CharsetRecog_2022 {
public:
    virtual ~CharsetRecog_2022KR();

    const char *getName() const override;

    UBool match(InputText *textIn, CharsetMatch *results) const override;

};

class CharsetRecog_2022CN :public CharsetRecog_2022
{
public:
    virtual ~CharsetRecog_2022CN();

    const char* getName() const override;

    UBool match(InputText *textIn, CharsetMatch *results) const override;
};
#endif

U_NAMESPACE_END

#endif
#endif /* __CSR2022_H */
