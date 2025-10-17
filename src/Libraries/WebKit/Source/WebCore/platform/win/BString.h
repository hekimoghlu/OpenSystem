/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#ifndef BString_h
#define BString_h

#include <wtf/Forward.h>

#ifndef _PREFAST_
typedef wchar_t* BSTR;
#else // _PREFAST_
typedef _Null_terminated_ wchar_t* BSTR;
#endif

namespace WebCore {

    class BString {
    public:
        WEBCORE_EXPORT BString();
        WEBCORE_EXPORT BString(const wchar_t*);
        WEBCORE_EXPORT BString(const wchar_t*, size_t length);
        WEBCORE_EXPORT BString(const String&);
        WEBCORE_EXPORT BString(const AtomString&);
        WEBCORE_EXPORT BString(const URL&);
        WEBCORE_EXPORT ~BString();

        WEBCORE_EXPORT void adoptBSTR(BSTR);
        WEBCORE_EXPORT void clear();

        WEBCORE_EXPORT BString(const BString&);
        BString& operator=(const BString&);
        WEBCORE_EXPORT BString& operator=(const BSTR&);

        BSTR* operator&() { ASSERT(!m_bstr); return &m_bstr; }
        operator BSTR() const { return m_bstr; }

        BSTR release() { BSTR result = m_bstr; m_bstr = 0; return result; }

    private:
        BSTR m_bstr;
    };

    WEBCORE_EXPORT bool operator ==(const BString&, const BString&);
    bool operator !=(const BString&, const BString&);
    bool operator ==(const BString&, BSTR);
    WEBCORE_EXPORT bool operator !=(const BString&, BSTR);
    bool operator ==(BSTR, const BString&);
    bool operator !=(BSTR, const BString&);

}

#endif
