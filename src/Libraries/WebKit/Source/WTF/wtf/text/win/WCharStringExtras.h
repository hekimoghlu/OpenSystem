/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#pragma once

#include <unicode/utypes.h>

namespace WTF {

inline const UChar* ucharFrom(const wchar_t* characters)
{
    static_assert(sizeof(wchar_t) == sizeof(UChar), "We assume wchar_t and UChar have the same size");
    return reinterpret_cast<const UChar*>(characters);
}

inline UChar* ucharFrom(wchar_t* characters)
{
    static_assert(sizeof(wchar_t) == sizeof(UChar), "We assume wchar_t and UChar have the same size");
    return reinterpret_cast<UChar*>(characters);
}

inline const wchar_t* wcharFrom(const UChar* characters)
{
    static_assert(sizeof(wchar_t) == sizeof(UChar), "We assume wchar_t and UChar have the same size");
    return reinterpret_cast<const wchar_t*>(characters);
}

inline wchar_t* wcharFrom(UChar* characters)
{
    static_assert(sizeof(wchar_t) == sizeof(UChar), "We assume wchar_t and UChar have the same size");
    return reinterpret_cast<wchar_t*>(characters);
}

} // namespace WTF

using WTF::ucharFrom;
using WTF::wcharFrom;
