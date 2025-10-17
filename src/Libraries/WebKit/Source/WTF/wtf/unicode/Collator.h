/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include <unicode/uconfig.h>
#include <wtf/Noncopyable.h>

struct UCharIterator;
struct UCollator;

namespace WTF {

class StringView;

#if UCONFIG_NO_COLLATION

class Collator {
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit Collator(const char* = nullptr, bool = false) { }

    WTF_EXPORT_PRIVATE static int collate(StringView, StringView);
    WTF_EXPORT_PRIVATE static int collate(const char8_t*, const char8_t*);
};

#else

class Collator {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(Collator);
public:
    // The value nullptr is a special one meaning the system default locale.
    // Locale name parsing is lenient; e.g. language identifiers (such as "en-US") are accepted, too.
    WTF_EXPORT_PRIVATE explicit Collator(const char* locale = nullptr, bool shouldSortLowercaseFirst = false);
    WTF_EXPORT_PRIVATE ~Collator();

    WTF_EXPORT_PRIVATE int collate(StringView, StringView) const;
    WTF_EXPORT_PRIVATE int collate(const char8_t*, const char8_t*) const;

private:
    char* m_locale;
    bool m_shouldSortLowercaseFirst;
    UCollator* m_collator;
};

WTF_EXPORT_PRIVATE UCharIterator createIterator(StringView);

#endif

}

using WTF::Collator;
