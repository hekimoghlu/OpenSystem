/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#include "PrintInfo.h"
#include <WebCore/SharedBuffer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/FastMalloc.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>

#if USE(CAIRO)
#include <WebCore/RefPtrCairo.h>
#elif USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkCanvas.h>
#include <skia/core/SkDocument.h>
#include <skia/core/SkPicture.h>
#include <skia/core/SkPictureRecorder.h>
#include <skia/core/SkStream.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif

typedef struct _GtkPrintSettings GtkPrintSettings;
typedef struct _GtkPageSetup GtkPageSetup;
typedef struct _GtkPageRange GtkPageRange;

namespace WebCore {
class PrintContext;
class ResourceError;
};

namespace WebKit {

class WebPrintOperationGtk {
    WTF_MAKE_TZONE_ALLOCATED(WebPrintOperationGtk);
public:
    explicit WebPrintOperationGtk(const PrintInfo&);
    ~WebPrintOperationGtk();

    void startPrint(WebCore::PrintContext*, CompletionHandler<void(RefPtr<WebCore::FragmentedSharedBuffer>&&, WebCore::ResourceError&&)>&&);

private:
#if USE(CAIRO)
    void startPage(cairo_t*);
    void endPage(cairo_t*);
    void endPrint(cairo_t*);
#elif USE(SKIA)
    void startPage(SkPictureRecorder&);
    void endPage(SkPictureRecorder&);
    void endPrint();
#endif

    struct PrintPagesData {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        explicit PrintPagesData(WebPrintOperationGtk*);

        size_t collatedCopiesLeft() const { return collatedCopies > 1 ? collatedCopies - collated - 1 : 0; }
        size_t uncollatedCopiesLeft() const { return uncollatedCopies > 1 ? uncollatedCopies - uncollated - 1 : 0; }
        size_t copiesLeft() const { return collatedCopiesLeft() + uncollatedCopiesLeft(); }

        void incrementPageSequence();

        WebPrintOperationGtk* printOperation { nullptr };
        GRefPtr<GMainLoop> mainLoop;
        int totalPrinted { -1 };
        int pageNumber { 0 };
        Vector<size_t> pages;
        size_t sheetNumber { 0 };
        size_t firstSheetNumber { 0 };
        size_t numberOfSheets { 0 };
        size_t firstPagePosition { 0 };
        size_t lastPagePosition { 0 };
        size_t collated { 0 };
        size_t uncollated { 0 };
        size_t collatedCopies { 0 };
        size_t uncollatedCopies { 0 };
        bool isDone { false };
        bool isValid { true };

    };

    static gboolean printPagesIdle(gpointer);
    static void printPagesIdleDone(gpointer);

    int pageCount() const;
    bool currentPageIsFirstPageOfSheet() const;
    bool currentPageIsLastPageOfSheet() const;
#if USE(CAIRO)
    void print(cairo_surface_t*, double xDPI, double yDPI);
#elif USE(SKIA)
    void print(double xDPI, double yDPI);
#endif
    void renderPage(int pageNumber);
    void rotatePageIfNeeded();
    void getRowsAndColumnsOfPagesPerSheet(size_t& rows, size_t& columns);
    void getPositionOfPageInSheet(size_t rows, size_t columns, int& x, int&y);
    void prepareContextToDraw();
    void printPagesDone();
    void printDone(RefPtr<WebCore::FragmentedSharedBuffer>&&, WebCore::ResourceError&&);
    URL frameURL() const;

    GRefPtr<GtkPrintSettings> m_printSettings;
    GRefPtr<GtkPageSetup> m_pageSetup;
    PrintInfo::PrintMode m_printMode { PrintInfo::PrintMode::Async };
    WebCore::PrintContext* m_printContext { nullptr };
    CompletionHandler<void(RefPtr<WebCore::FragmentedSharedBuffer>&&, WebCore::ResourceError&&)> m_completionHandler;
    double m_xDPI { 1 };
    double m_yDPI { 1 };

#if USE(CAIRO)
    WebCore::SharedBufferBuilder m_buffer;
    RefPtr<cairo_t> m_cairoContext;
#elif USE(SKIA)
    Vector<sk_sp<SkPicture>> m_pages;
    SkCanvas* m_pageCanvas { nullptr };
#endif

    unsigned m_printPagesIdleId { 0 };
    size_t m_numberOfPagesToPrint { 0 };
    unsigned m_pagesToPrint { 0 };
    size_t m_pagePosition { 0 };
    GtkPageRange* m_pageRanges { nullptr };
    size_t m_pageRangesCount { 0 };
    bool m_needsRotation { false };

    // Manual capabilities.
    unsigned m_numberUp { 1 };
    unsigned m_numberUpLayout { 0 };
    unsigned m_pageSet { 0 };
    bool m_reverse { false };
    unsigned m_copies { 1 };
    bool m_collateCopies { false };
    double m_scale { 1 };
};

} // namespace WebKit
