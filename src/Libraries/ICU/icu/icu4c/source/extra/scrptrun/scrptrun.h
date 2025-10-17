/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
 *   Copyright (C) 1999-2003, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *
 *******************************************************************************
 *   file name:  scrptrun.h
 *
 *   created on: 10/17/2001
 *   created by: Eric R. Mader
 */

#ifndef __SCRPTRUN_H
#define __SCRPTRUN_H

#include "unicode/utypes.h"
#include "unicode/uobject.h"
#include "unicode/uscript.h"

U_NAMESPACE_BEGIN

struct ScriptRecord
{
    UChar32 startChar;
    UChar32 endChar;
    UScriptCode scriptCode;
};

struct ParenStackEntry
{
    int32_t pairIndex;
    UScriptCode scriptCode;
};

class ScriptRun : public UObject {
public:
    ScriptRun();

    ScriptRun(const char16_t chars[], int32_t length);

    ScriptRun(const char16_t chars[], int32_t start, int32_t length);

    void reset();

    void reset(int32_t start, int32_t count);

    void reset(const char16_t chars[], int32_t start, int32_t length);

    int32_t getScriptStart();

    int32_t getScriptEnd();

    UScriptCode getScriptCode();

    UBool next();

    /**
     * ICU "poor man's RTTI", returns a UClassID for the actual class.
     *
     * @stable ICU 2.2
     */
    virtual inline UClassID getDynamicClassID() const override { return getStaticClassID(); }

    /**
     * ICU "poor man's RTTI", returns a UClassID for this class.
     *
     * @stable ICU 2.2
     */
    static inline UClassID getStaticClassID() { return (UClassID)&fgClassID; }

private:

    static UBool sameScript(int32_t scriptOne, int32_t scriptTwo);

    int32_t charStart;
    int32_t charLimit;
    const char16_t *charArray;

    int32_t scriptStart;
    int32_t scriptEnd;
    UScriptCode scriptCode;

    ParenStackEntry parenStack[128];
    int32_t parenSP;

    static int8_t highBit(int32_t value);
    static int32_t getPairIndex(UChar32 ch);

    static UChar32 pairedChars[];
    static const int32_t pairedCharCount;
    static const int32_t pairedCharPower;
    static const int32_t pairedCharExtra;

    /**
     * The address of this static class variable serves as this class's ID
     * for ICU "poor man's RTTI".
     */
    static const char fgClassID;
};

inline ScriptRun::ScriptRun()
{
    reset(nullptr, 0, 0);
}

inline ScriptRun::ScriptRun(const char16_t chars[], int32_t length)
{
    reset(chars, 0, length);
}

inline ScriptRun::ScriptRun(const char16_t chars[], int32_t start, int32_t length)
{
    reset(chars, start, length);
}

inline int32_t ScriptRun::getScriptStart()
{
    return scriptStart;
}

inline int32_t ScriptRun::getScriptEnd()
{
    return scriptEnd;
}

inline UScriptCode ScriptRun::getScriptCode()
{
    return scriptCode;
}

inline void ScriptRun::reset()
{
    scriptStart = charStart;
    scriptEnd   = charStart;
    scriptCode  = USCRIPT_INVALID_CODE;
    parenSP     = -1;
}

inline void ScriptRun::reset(int32_t start, int32_t length)
{
    charStart = start;
    charLimit = start + length;

    reset();
}

inline void ScriptRun::reset(const char16_t chars[], int32_t start, int32_t length)
{
    charArray = chars;

    reset(start, length);
}

U_NAMESPACE_END

#endif
