/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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

#include "ArgumentCoders.h"
#include <variant>
#include <wtf/HashTraits.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

struct WebFoundTextRange {
    struct DOMData {
        uint64_t location { 0 };
        uint64_t length { 0 };

        bool operator==(const DOMData& other) const = default;
    };

    struct PDFData {
        uint64_t startPage { 0 };
        uint64_t startOffset { 0 };
        uint64_t endPage { 0 };
        uint64_t endOffset { 0 };

        bool operator==(const PDFData& other) const = default;
    };

    std::variant<DOMData, PDFData> data { DOMData { } };
    AtomString frameIdentifier;
    uint64_t order { 0 };

    unsigned hash() const;

    bool operator==(const WebFoundTextRange& other) const;
};

} // namespace WebKit

namespace WTF {

struct WebFoundTextRangeHash {
    static unsigned hash(const WebKit::WebFoundTextRange& range) { return range.hash(); }
    static bool equal(const WebKit::WebFoundTextRange& a, const WebKit::WebFoundTextRange& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebKit::WebFoundTextRange> : GenericHashTraits<WebKit::WebFoundTextRange> {
    static WebKit::WebFoundTextRange emptyValue() { return { }; }

    static void constructDeletedValue(WebKit::WebFoundTextRange& slot) { new (NotNull, &slot.frameIdentifier) AtomString { HashTableDeletedValue }; }
    static bool isDeletedValue(const WebKit::WebFoundTextRange& range) { return range.frameIdentifier.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebKit::WebFoundTextRange> : WebFoundTextRangeHash { };

}
