/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

#if HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)

#include "BarcodeDetectorInterface.h"
#include <wtf/HashFunctions.h>
#include <wtf/HashSet.h>
#include <wtf/HashTraits.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::ShapeDetection {

struct BarcodeDetectorOptions;

class BarcodeDetectorImpl final : public BarcodeDetector {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(BarcodeDetectorImpl, WEBCORE_EXPORT);
public:
    static Ref<BarcodeDetectorImpl> create(const BarcodeDetectorOptions& barcodeDetectorOptions)
    {
        return adoptRef(*new BarcodeDetectorImpl(barcodeDetectorOptions));
    }

    virtual ~BarcodeDetectorImpl();

    WEBCORE_EXPORT static void getSupportedFormats(CompletionHandler<void(Vector<BarcodeFormat>&&)>&&);

    using BarcodeFormatSet = HashSet<BarcodeFormat, WTF::IntHash<BarcodeFormat>, WTF::StrongEnumHashTraits<BarcodeFormat>>;

private:
    WEBCORE_EXPORT BarcodeDetectorImpl(const BarcodeDetectorOptions&);

    BarcodeDetectorImpl(const BarcodeDetectorImpl&) = delete;
    BarcodeDetectorImpl(BarcodeDetectorImpl&&) = delete;
    BarcodeDetectorImpl& operator=(const BarcodeDetectorImpl&) = delete;
    BarcodeDetectorImpl& operator=(BarcodeDetectorImpl&&) = delete;

    void detect(Ref<ImageBuffer>&&, CompletionHandler<void(Vector<DetectedBarcode>&&)>&&) final;

    std::optional<BarcodeFormatSet> m_requestedBarcodeFormatSet;
};

} // namespace WebCore::ShapeDetection

#endif // HAVE(SHAPE_DETECTION_API_IMPLEMENTATION) && HAVE(VISION)
