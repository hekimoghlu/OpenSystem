/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#include "config.h"
#include "BString.h"

#include <windows.h>
#include <wtf/URL.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/StringView.h>
#include <wtf/text/win/WCharStringExtras.h>

namespace WebCore {
using namespace JSC;

BString::BString()
    : m_bstr(0)
{
}

BString::BString(const wchar_t* characters)
{
    if (!characters)
        m_bstr = 0;
    else
        m_bstr = SysAllocString(characters);
}

BString::BString(const wchar_t* characters, size_t length)
{
    if (!characters)
        m_bstr = 0;
    else
        m_bstr = SysAllocStringLen(characters, length);
}

BString::BString(const String& s)
{
    if (s.isNull())
        m_bstr = 0;
    else
        m_bstr = SysAllocStringLen(wcharFrom(StringView(s).upconvertedCharacters()), s.length());
}

BString::BString(const URL& url)
{
    if (url.isNull())
        m_bstr = 0;
    else
        m_bstr = SysAllocStringLen(wcharFrom(StringView(url.string()).upconvertedCharacters()), url.string().length());
}

BString::BString(const AtomString& s)
{
    if (s.isNull())
        m_bstr = 0;
    else
        m_bstr = SysAllocStringLen(wcharFrom(StringView(s.string()).upconvertedCharacters()), s.length());
}

BString::~BString()
{
    SysFreeString(m_bstr);
}

BString::BString(const BString& other)
{
    if (!other.m_bstr)
        m_bstr = 0;
    else
        m_bstr = SysAllocString(other.m_bstr);
}

void BString::adoptBSTR(BSTR bstr)
{
    SysFreeString(m_bstr);
    m_bstr = bstr;
}

void BString::clear()
{
    SysFreeString(m_bstr);
    m_bstr = 0;
}

BString& BString::operator=(const BString& other)
{
    if (this != &other)
        *this = other.m_bstr;
    return *this;
}

BString& BString::operator=(const BSTR& other)
{
    if (other != m_bstr) {
        SysFreeString(m_bstr);
        m_bstr = other ? SysAllocString(other) : 0;
    }

    return *this;
}

bool operator ==(const BString& a, const BString& b)
{
    if (SysStringLen((BSTR)a) != SysStringLen((BSTR)b))
        return false;
    if (!(BSTR)a && !(BSTR)b)
        return true;
    if (!(BSTR)a || !(BSTR)b)
        return false;
    return !wcscmp((BSTR)a, (BSTR)b);
}

bool operator !=(const BString& a, const BString& b)
{
    return !(a==b);
}

bool operator ==(const BString& a, BSTR b)
{
    if (SysStringLen((BSTR)a) != SysStringLen(b))
        return false;
    if (!(BSTR)a && !b)
        return true;
    if (!(BSTR)a || !b)
        return false;
    return !wcscmp((BSTR)a, b);
}

bool operator !=(const BString& a, BSTR b)
{
    return !(a==b);
}

bool operator ==(BSTR a, const BString& b)
{
    if (SysStringLen(a) != SysStringLen((BSTR)b))
        return false;
    if (!a && !(BSTR)b)
        return true;
    if (!a || !(BSTR)b)
        return false;
    return !wcscmp(a, (BSTR)b);
}

bool operator !=(BSTR a, const BString& b)
{
    return !(a==b);
}

}
