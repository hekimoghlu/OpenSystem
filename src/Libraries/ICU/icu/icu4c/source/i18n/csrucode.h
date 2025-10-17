/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#ifndef __CSRUCODE_H
#define __CSRUCODE_H

#include "unicode/utypes.h"

#if !UCONFIG_NO_CONVERSION

#include "csrecog.h"

U_NAMESPACE_BEGIN

/**
 * This class matches UTF-16 and UTF-32, both big- and little-endian. The
 * BOM will be used if it is present.
 * 
 * @internal
 */
class CharsetRecog_Unicode : public CharsetRecognizer 
{

public:

    virtual ~CharsetRecog_Unicode();
    /* (non-Javadoc)
     * @see com.ibm.icu.text.CharsetRecognizer#getName()
     */
    const char* getName() const override = 0;

    /* (non-Javadoc)
     * @see com.ibm.icu.text.CharsetRecognizer#match(com.ibm.icu.text.CharsetDetector)
     */
    UBool match(InputText* textIn, CharsetMatch *results) const override = 0;
};


class CharsetRecog_UTF_16_BE : public CharsetRecog_Unicode
{
public:

    virtual ~CharsetRecog_UTF_16_BE();

    const char *getName() const override;

    UBool match(InputText* textIn, CharsetMatch *results) const override;
};

class CharsetRecog_UTF_16_LE : public CharsetRecog_Unicode
{
public:

    virtual ~CharsetRecog_UTF_16_LE();

    const char *getName() const override;

    UBool match(InputText* textIn, CharsetMatch *results) const override;
};

class CharsetRecog_UTF_32 : public CharsetRecog_Unicode
{
protected:
    virtual int32_t getChar(const uint8_t *input, int32_t index) const = 0;
public:

    virtual ~CharsetRecog_UTF_32();

    const char* getName() const override = 0;

    UBool match(InputText* textIn, CharsetMatch *results) const override;
};


class CharsetRecog_UTF_32_BE : public CharsetRecog_UTF_32
{
protected:
    int32_t getChar(const uint8_t *input, int32_t index) const override;

public:

    virtual ~CharsetRecog_UTF_32_BE();

    const char *getName() const override;
};


class CharsetRecog_UTF_32_LE : public CharsetRecog_UTF_32
{
protected:
    int32_t getChar(const uint8_t *input, int32_t index) const override;

public:
    virtual ~CharsetRecog_UTF_32_LE();

    const char* getName() const override;
};

U_NAMESPACE_END

#endif
#endif /* __CSRUCODE_H */
