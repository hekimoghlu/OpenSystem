/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
#include <wtf/URL.h>

#include <CoreFoundation/CFURL.h>
#include <wtf/URLParser.h>
#include <wtf/cf/CFURLExtras.h>
#include <wtf/text/CString.h>

namespace WTF {

URL::URL(CFURLRef url)
{
    // FIXME: Why is it OK to ignore the base URL in the CFURL here?
    if (!url)
        invalidate();
    else
        *this = URLParser(bytesAsString(url)).result();
}

#if !USE(FOUNDATION)

RetainPtr<CFURLRef> URL::emptyCFURL()
{
    return nullptr;
}

#endif

RetainPtr<CFURLRef> URL::createCFURL() const
{
    if (isNull())
        return nullptr;

    if (isEmpty())
        return emptyCFURL();

    RetainPtr<CFURLRef> result;
    if (LIKELY(m_string.is8Bit() && m_string.containsOnlyASCII())) {
        auto characters = m_string.span8();
        result = adoptCF(CFURLCreateAbsoluteURLWithBytes(nullptr, characters.data(), characters.size(), kCFStringEncodingUTF8, nullptr, true));
    } else {
        CString utf8 = m_string.utf8();
        auto utf8Span = utf8.span();
        result = adoptCF(CFURLCreateAbsoluteURLWithBytes(nullptr, utf8Span.data(), utf8Span.size(), kCFStringEncodingUTF8, nullptr, true));
    }

    if (protocolIsInHTTPFamily() && !isSameOrigin(result.get(), *this))
        return nullptr;

    return result;
}

#if !PLATFORM(WIN)
String URL::fileSystemPath() const
{
    auto cfURL = createCFURL();
    if (!cfURL)
        return String();

    return adoptCF(CFURLCopyFileSystemPath(cfURL.get(), kCFURLPOSIXPathStyle)).get();
}
#endif

}
