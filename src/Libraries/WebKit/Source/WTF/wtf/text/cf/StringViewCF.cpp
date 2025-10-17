/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include <wtf/text/StringView.h>

#if USE(CF)

#include <CoreFoundation/CoreFoundation.h>
#include <wtf/RetainPtr.h>

namespace WTF {

RetainPtr<CFStringRef> StringView::createCFString() const
{
    if (is8Bit()) {
        auto characters = span8();
        return adoptCF(CFStringCreateWithBytes(kCFAllocatorDefault, characters.data(), characters.size(), kCFStringEncodingISOLatin1, false));
    }
    auto characters = span16();
    return adoptCF(CFStringCreateWithCharacters(kCFAllocatorDefault, reinterpret_cast<const UniChar*>(characters.data()), characters.size()));
}

RetainPtr<CFStringRef> StringView::createCFStringWithoutCopying() const
{
    if (is8Bit()) {
        auto characters = span8();
        return adoptCF(CFStringCreateWithBytesNoCopy(kCFAllocatorDefault, characters.data(), characters.size(), kCFStringEncodingISOLatin1, false, kCFAllocatorNull));
    }
    auto characters = span16();
    return adoptCF(CFStringCreateWithCharactersNoCopy(kCFAllocatorDefault, reinterpret_cast<const UniChar*>(characters.data()), characters.size(), kCFAllocatorNull));
}

}

#endif // USE(CF)
