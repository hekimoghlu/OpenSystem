/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#if ENABLE(UNIFIED_PDF)

#include "PDFDocumentLayout.h"
#include <wtf/text/ParsingUtilities.h>

namespace WTF {
class TextStream;
}

namespace WebKit {

struct PerPageInfo {
    PDFDocumentLayout::PageIndex pageIndex { 0 };
    WebCore::FloatRect pageBounds; // These are in "PDFDocumentLayout" coordinates.
    WebCore::FloatRect rectInPageLayoutCoordinates; // Some arbirary rect converted into "PDFDocumentLayout" coordinates.

    bool operator==(const PerPageInfo&) const = default;
};

using PDFPageCoverage = Vector<PerPageInfo>;

inline PerPageInfo unite(const PerPageInfo& a, const PerPageInfo& b)
{
    return { a.pageIndex, a.pageBounds, unionRect(a.rectInPageLayoutCoordinates, b.rectInPageLayoutCoordinates) };
}

inline PDFPageCoverage unite(const PDFPageCoverage& a, const PDFPageCoverage& b)
{
    PDFPageCoverage result;
    result.reserveCapacity(a.size() + b.size());
    auto as = a.span();
    auto bs = b.span();
    while (!as.empty() && !bs.empty()) {
        auto cmp = as.front().pageIndex <=> bs.front().pageIndex;
        if (cmp < 0)
            result.append(consume(as));
        else if (cmp > 0)
            result.append(consume(bs));
        else
            result.append(unite(consume(as), consume(bs)));
    }
    result.append(as);
    result.append(bs);
    return result;
}

struct PDFPageCoverageAndScales {
    PDFPageCoverage pages;
    WebCore::FloatSize contentsOffset { };
    float deviceScaleFactor { 1 };
    float pdfDocumentScale { 1 };
    float tilingScaleFactor { 1 };

    bool operator==(const PDFPageCoverageAndScales&) const = default;
};

WTF::TextStream& operator<<(WTF::TextStream&, const PerPageInfo&);
WTF::TextStream& operator<<(WTF::TextStream&, const PDFPageCoverageAndScales&);

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
