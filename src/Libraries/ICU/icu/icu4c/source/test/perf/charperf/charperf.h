/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
#ifndef _CHARPERF_H
#define _CHARPERF_H

#include "unicode/uchar.h"

#include "unicode/uperf.h"
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>
#include <wctype.h>

typedef void (*CharPerfFn)(UChar32 ch);
typedef void (*StdLibCharPerfFn)(wchar_t ch);

class CharPerfFunction : public UPerfFunction
{
public:
    void call(UErrorCode* status) override
    {
        for (UChar32 i = MIN_; i < MAX_; i ++) {
            (*m_fn_)(i);
        }
    }

    long getOperationsPerIteration() override
    {
        return MAX_ - MIN_;
    }
    CharPerfFunction(CharPerfFn func, UChar32 min, UChar32 max)
    {
        m_fn_ = func;
        MIN_ = min;
        MAX_ = max;
    }   

private:
    CharPerfFn m_fn_;
    UChar32 MIN_;
    UChar32 MAX_;
}; 

class StdLibCharPerfFunction : public UPerfFunction
{
public:
    void call(UErrorCode* status) override
    {
        // note wchar_t is unsigned, it will revert to 0 once it reaches 
        // 65535
        for (wchar_t i = MIN_; i < MAX_; i ++) {
            (*m_fn_)(i);
        }
    }

    long getOperationsPerIteration() override
    {
        return MAX_ - MIN_;
    }

    StdLibCharPerfFunction(StdLibCharPerfFn func, wchar_t min, wchar_t max)
    {
        m_fn_ = func;			
        MIN_ = min;
        MAX_ = max;
    }   

    ~StdLibCharPerfFunction()
    {			
    }

private:
    StdLibCharPerfFn m_fn_;
    wchar_t MIN_;
    wchar_t MAX_;
};

class CharPerformanceTest : public UPerfTest
{
public:
    CharPerformanceTest(int32_t argc, const char *argv[], UErrorCode &status);
    ~CharPerformanceTest();
    UPerfFunction* runIndexedTest(int32_t index, UBool exec,
        const char*& name,
        char* par = nullptr) override;
    UPerfFunction* TestIsAlpha();
    UPerfFunction* TestIsUpper();
    UPerfFunction* TestIsLower();
    UPerfFunction* TestIsDigit();
    UPerfFunction* TestIsSpace();
    UPerfFunction* TestIsAlphaNumeric();
    UPerfFunction* TestIsPrint();
    UPerfFunction* TestIsControl();
    UPerfFunction* TestToLower();
    UPerfFunction* TestToUpper();
    UPerfFunction* TestIsWhiteSpace();
    UPerfFunction* TestStdLibIsAlpha();
    UPerfFunction* TestStdLibIsUpper();
    UPerfFunction* TestStdLibIsLower();
    UPerfFunction* TestStdLibIsDigit();
    UPerfFunction* TestStdLibIsSpace();
    UPerfFunction* TestStdLibIsAlphaNumeric();
    UPerfFunction* TestStdLibIsPrint();
    UPerfFunction* TestStdLibIsControl();
    UPerfFunction* TestStdLibToLower();
    UPerfFunction* TestStdLibToUpper();
    UPerfFunction* TestStdLibIsWhiteSpace();

private:
    UChar32 MIN_;
    UChar32 MAX_;
};

inline void isAlpha(UChar32 ch) 
{
    u_isalpha(ch);
}

inline void isUpper(UChar32 ch)
{
    u_isupper(ch);
}

inline void isLower(UChar32 ch)
{
    u_islower(ch);
}

inline void isDigit(UChar32 ch)
{
    u_isdigit(ch);
}

inline void isSpace(UChar32 ch)
{
    u_isspace(ch);
}

inline void isAlphaNumeric(UChar32 ch)
{
    u_isalnum(ch);
}

/**
* This test may be different since c lib has a type PUNCT and it is printable.
* iswgraph is not used for testing since it is a subset of iswprint with the
* exception of returning true for white spaces. no match found in icu4c.
*/
inline void isPrint(UChar32 ch)
{
    u_isprint(ch);
}

inline void isControl(UChar32 ch)
{
    u_iscntrl(ch);
}

inline void toLower(UChar32 ch)
{
    u_tolower(ch);
}

inline void toUpper(UChar32 ch)
{
    u_toupper(ch);
}

inline void isWhiteSpace(UChar32 ch)
{
    u_isWhitespace(ch);
}

inline void StdLibIsAlpha(wchar_t ch)
{
    iswalpha(ch);
}

inline void StdLibIsUpper(wchar_t ch)
{
    iswupper(ch);
}

inline void StdLibIsLower(wchar_t ch)
{
    iswlower(ch);
}

inline void StdLibIsDigit(wchar_t ch)
{
    iswdigit(ch);
}

inline void StdLibIsSpace(wchar_t ch)
{
    iswspace(ch);
}

inline void StdLibIsAlphaNumeric(wchar_t ch)
{
    iswalnum(ch);
}

/**
* This test may be different since c lib has a type PUNCT and it is printable.
* iswgraph is not used for testing since it is a subset of iswprint with the
* exception of returning true for white spaces. no match found in icu4c.
*/
inline void StdLibIsPrint(wchar_t ch)
{
    iswprint(ch);
}

inline void StdLibIsControl(wchar_t ch)
{
    iswcntrl(ch);
}

inline void StdLibToLower(wchar_t ch)
{
    towlower(ch);
}

inline void StdLibToUpper(wchar_t ch)
{
    towupper(ch);
}

inline void StdLibIsWhiteSpace(wchar_t ch)
{
    iswspace(ch);
}

#endif // CHARPERF_H
