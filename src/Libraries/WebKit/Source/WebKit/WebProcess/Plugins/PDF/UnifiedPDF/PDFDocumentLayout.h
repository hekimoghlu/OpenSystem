/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

#include <WebCore/FloatRect.h>
#include <WebCore/IntDegrees.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS PDFDocument;
OBJC_CLASS PDFPage;

namespace WebCore {
class GraphicsContext;
class IntRect;
}

namespace WebKit {

enum class ShouldUpdateAutoSizeScale : bool { No, Yes };

struct PDFLayoutRow;

class PDFDocumentLayout {
public:
    using PageIndex = size_t; // This is a zero-based index.

    enum class DisplayMode : uint8_t {
        SinglePageDiscrete,
        SinglePageContinuous,
        TwoUpDiscrete,
        TwoUpContinuous,
    };

    PDFDocumentLayout();
    ~PDFDocumentLayout();

    void setPDFDocument(PDFDocument *document) { m_pdfDocument = document; }
    bool hasPDFDocument() const { return !!m_pdfDocument; }
    bool hasLaidOutPDFDocument() const { return !m_pageGeometry.isEmpty(); }

    size_t pageCount() const;
    size_t rowCount() const;
    PDFLayoutRow rowForPageIndex(PageIndex) const;
    Vector<PDFLayoutRow> rows() const;
    unsigned rowIndexForPageIndex(PageIndex) const;

    static constexpr WebCore::FloatSize documentMargin { 16, 18 };
    static constexpr WebCore::FloatSize pageMargin { 14, 16 };

    bool isLeftPageIndex(PageIndex) const;
    bool isRightPageIndex(PageIndex) const;
    bool isLastPageIndex(PageIndex) const;
    PageIndex lastPageIndex() const;
    bool isFirstPageOfRow(PageIndex) const;

    RetainPtr<PDFPage> pageAtIndex(PageIndex) const;
    std::optional<PageIndex> indexForPage(RetainPtr<PDFPage>) const;
    PageIndex nearestPageIndexForDocumentPoint(WebCore::FloatPoint, const std::optional<PDFLayoutRow>&) const;

    // For the given Y offset, return a page index and page point for the page at this offset. Returns the leftmost
    // page if two-up and both pages intersect that offset, otherwise the right page if only it intersects the offset.
    // The point is centered horizontally in the given page.
    // FIXME: <https://webkit.org/b/276981> Remove or provide row.
    std::pair<PageIndex, WebCore::FloatPoint> pageIndexAndPagePointForDocumentYOffset(float) const;

    // This is not scaled by scale().
    WebCore::FloatRect layoutBoundsForPageAtIndex(PageIndex) const;
    // Bounds of the pages in the row, including document margins. Not scaled.
    WebCore::FloatRect layoutBoundsForRow(PDFLayoutRow) const;

    // Returns 0, 90, 180, 270.
    WebCore::IntDegrees rotationForPageAtIndex(PageIndex) const;

    WebCore::FloatPoint documentToPDFPage(WebCore::FloatPoint documentPoint, PageIndex) const;
    WebCore::FloatRect documentToPDFPage(WebCore::FloatRect documentRect, PageIndex) const;

    WebCore::FloatPoint pdfPageToDocument(WebCore::FloatPoint pagePoint, PageIndex) const;
    WebCore::FloatRect pdfPageToDocument(WebCore::FloatRect pageRect, PageIndex) const;

    // This is the scale that scales the largest page or pair of pages up or down to fit the available width.
    float scale() const { return m_scale; }

    enum class LayoutUpdateChange : uint8_t {
        PageGeometries = 1 << 0,
        DocumentBounds = 1 << 1,
    };

    OptionSet<LayoutUpdateChange> updateLayout(WebCore::IntSize pluginSize, ShouldUpdateAutoSizeScale);
    WebCore::FloatSize contentsSize() const;
    WebCore::FloatSize scaledContentsSize() const;

    void setDisplayMode(DisplayMode displayMode) { m_displayMode = displayMode; }
    DisplayMode displayMode() const { return m_displayMode; }

    constexpr static bool isSinglePageDisplayMode(DisplayMode mode) { return mode == DisplayMode::SinglePageDiscrete || mode == DisplayMode::SinglePageContinuous; }
    constexpr static bool isTwoUpDisplayMode(DisplayMode mode) { return mode == DisplayMode::TwoUpDiscrete || mode == DisplayMode::TwoUpContinuous; }

    constexpr static bool isScrollingDisplayMode(DisplayMode mode) { return mode == DisplayMode::SinglePageContinuous || mode == DisplayMode::TwoUpContinuous; }
    constexpr static bool isDiscreteDisplayMode(DisplayMode mode) { return mode == DisplayMode::SinglePageDiscrete || mode == DisplayMode::TwoUpDiscrete; }

    bool isSinglePageDisplayMode() const { return isSinglePageDisplayMode(m_displayMode); }
    bool isTwoUpDisplayMode() const { return isTwoUpDisplayMode(m_displayMode); }

    bool isScrollingDisplayMode() const { return isScrollingDisplayMode(m_displayMode); }
    bool isDiscreteDisplayMode() const { return isDiscreteDisplayMode(m_displayMode); }

    unsigned pagesPerRow() const { return isSinglePageDisplayMode() ? 1 : 2; }

    struct PageGeometry {
        WebCore::FloatRect cropBox;
        WebCore::FloatRect layoutBounds;
        WebCore::IntDegrees rotation { 0 };

        friend bool operator==(const PageGeometry&, const PageGeometry&) = default;
    };

    std::optional<PageGeometry> geometryForPage(RetainPtr<PDFPage>) const;
    WebCore::AffineTransform toPageTransform(const PageGeometry&) const;

private:
    void layoutPages(WebCore::FloatSize availableSize, WebCore::FloatSize maxRowSize, ShouldUpdateAutoSizeScale);

    // maxRowSize does not include document margins.
    enum class CenterRowVertically : bool { No, Yes };
    void layoutContinuousRows(WebCore::FloatSize availableSize, WebCore::FloatSize maxRowSize, CenterRowVertically);
    void layoutDiscreteRows(WebCore::FloatSize availableSize, WebCore::FloatSize maxRowSize, CenterRowVertically);

    // Returns row rect including page margins.
    WebCore::FloatRect layoutRow(const PDFLayoutRow&, WebCore::FloatSize maxRowSize, float rowTop, CenterRowVertically centerVertically);

    RetainPtr<PDFDocument> m_pdfDocument;
    Vector<PageGeometry> m_pageGeometry;
    WebCore::FloatRect m_documentBounds;
    float m_scale { 1 };
    DisplayMode m_displayMode { DisplayMode::SinglePageContinuous };
};

struct PDFLayoutRow {
    Vector<PDFDocumentLayout::PageIndex, 2> pages;
    unsigned numPages() const { return pages.size(); }
    bool containsPage(PDFDocumentLayout::PageIndex pageIndex) const { return pages.contains(pageIndex); }
};

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
