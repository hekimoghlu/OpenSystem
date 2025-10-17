/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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

#if ENABLE(DATA_DETECTION)

#include "ImageOverlayDataDetectionResultIdentifier.h"
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMallocInlines.h>

OBJC_CLASS DDScannerResult;
OBJC_CLASS NSArray;

namespace WebCore {

class DataDetectionResultsStorage {
    WTF_MAKE_NONCOPYABLE(DataDetectionResultsStorage);
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DataDetectionResultsStorage);
public:
    DataDetectionResultsStorage() = default;

    void setDocumentLevelResults(NSArray *results) { m_documentLevelResults = results; }
    NSArray *documentLevelResults() const { return m_documentLevelResults.get(); }

    DDScannerResult *imageOverlayDataDetectionResult(ImageOverlayDataDetectionResultIdentifier identifier) { return m_imageOverlayResults.get(identifier).get(); }
    ImageOverlayDataDetectionResultIdentifier addImageOverlayDataDetectionResult(DDScannerResult *result)
    {
        auto newIdentifier = ImageOverlayDataDetectionResultIdentifier::generate();
        m_imageOverlayResults.set(newIdentifier, result);
        return newIdentifier;
    }

private:
    RetainPtr<NSArray> m_documentLevelResults;
    UncheckedKeyHashMap<ImageOverlayDataDetectionResultIdentifier, RetainPtr<DDScannerResult>> m_imageOverlayResults;
};

} // namespace WebCore

#endif // ENABLE(DATA_DETECTION)
