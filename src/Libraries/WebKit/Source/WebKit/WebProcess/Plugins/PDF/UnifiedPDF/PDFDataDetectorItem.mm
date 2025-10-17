/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#include "PDFDataDetectorItem.h"

#if ENABLE(UNIFIED_PDF_DATA_DETECTION)

#include "PDFKitSPI.h"
#include <pal/spi/cocoa/DataDetectorsCoreSPI.h>
#include <pal/spi/mac/DataDetectorsSPI.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/cf/TypeCastsCF.h>

#include "PDFKitSoftlink.h"
#include <pal/cocoa/DataDetectorsCoreSoftLink.h>
#include <pal/mac/DataDetectorsSoftLink.h>

namespace WebKit {

static bool hasActionsForResult(DDScannerResult *dataDetectorResult)
{
    return [[PAL::getDDActionsManagerClass() sharedManager] hasActionsForResult:[dataDetectorResult coreResult] actionContext:nil];
}

static bool resultIsPastDate(DDScannerResult *dataDetectorResult, PDFPage *pdfPage)
{
    NSDate *referenceDate = [[[pdfPage document] documentAttributes] objectForKey:get_PDFKit_PDFDocumentCreationDateAttribute()];
    RetainPtr referenceTimeZone = adoptCF(CFTimeZoneCopyDefault());
    return PAL::softLink_DataDetectorsCore_DDResultIsPastDate([dataDetectorResult coreResult], (CFDateRef)referenceDate, (CFTimeZoneRef)referenceTimeZone.get());
}

Ref<PDFDataDetectorItem> PDFDataDetectorItem::create(DDScannerResult *dataDetectorResult, PDFPage *pdfPage)
{
    return adoptRef(*new PDFDataDetectorItem(dataDetectorResult, pdfPage));
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(PDFDataDetectorItem);

PDFDataDetectorItem::PDFDataDetectorItem(DDScannerResult *dataDetectorResult, PDFPage *pdfPage)
    : m_result { dataDetectorResult }
    , m_page { pdfPage }
    , m_hasActions { hasActionsForResult(dataDetectorResult) }
    , m_isPastDate { resultIsPastDate(dataDetectorResult, pdfPage) }
{
}

DDScannerResult *PDFDataDetectorItem::scannerResult() const
{
    return m_result.get();
}

bool PDFDataDetectorItem::hasActions() const
{
    return m_hasActions;
}

bool PDFDataDetectorItem::isPastDate() const
{
    return m_isPastDate;
}

RetainPtr<PDFSelection> PDFDataDetectorItem::selection() const
{
    return [m_page selectionForRange:[m_result urlificationRange]];
}

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF_DATA_DETECTION)
