/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

#if ENABLE(UNIFIED_PDF_DATA_DETECTION)

#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS DDScannerResult;
OBJC_CLASS PDFPage;
OBJC_CLASS PDFSelection;

namespace WebCore {
class FloatRect;
}

namespace WebKit {

class PDFDataDetectorItem : public RefCounted<PDFDataDetectorItem> {
    WTF_MAKE_TZONE_ALLOCATED(PDFDataDetectorItem);
    WTF_MAKE_NONCOPYABLE(PDFDataDetectorItem);
public:
    static Ref<PDFDataDetectorItem> create(DDScannerResult *, PDFPage *);

    DDScannerResult *scannerResult() const;
    bool hasActions() const;
    bool isPastDate() const;
    RetainPtr<PDFSelection> selection() const;

private:
    PDFDataDetectorItem(DDScannerResult *, PDFPage *);

    RetainPtr<DDScannerResult> m_result;
    RetainPtr<PDFPage> m_page;
    bool m_hasActions { false };
    bool m_isPastDate { false };
};

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF_DATA_DETECTION)
