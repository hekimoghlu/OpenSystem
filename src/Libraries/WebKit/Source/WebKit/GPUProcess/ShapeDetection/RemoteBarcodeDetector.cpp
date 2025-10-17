/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#include "RemoteBarcodeDetector.h"

#if ENABLE(GPU_PROCESS)

#include "ArgumentCoders.h"
#include "RemoteRenderingBackend.h"
#include "RemoteResourceCache.h"
#include "ShapeDetectionObjectHeap.h"
#include "SharedPreferencesForWebProcess.h"
#include <WebCore/BarcodeDetectorInterface.h>
#include <WebCore/DetectedBarcodeInterface.h>
#include <WebCore/ImageBuffer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteBarcodeDetector);

RemoteBarcodeDetector::RemoteBarcodeDetector(Ref<WebCore::ShapeDetection::BarcodeDetector>&& barcodeDetector, ShapeDetection::ObjectHeap& objectHeap, RemoteRenderingBackend& backend, ShapeDetectionIdentifier identifier, WebCore::ProcessIdentifier webProcessIdentifier)
    : m_backing(WTFMove(barcodeDetector))
    , m_objectHeap(objectHeap)
    , m_backend(backend)
    , m_identifier(identifier)
    , m_webProcessIdentifier(webProcessIdentifier)
{
}

RemoteBarcodeDetector::~RemoteBarcodeDetector() = default;

std::optional<SharedPreferencesForWebProcess> RemoteBarcodeDetector::sharedPreferencesForWebProcess() const
{
    return protectedBackend()->sharedPreferencesForWebProcess();
}

Ref<WebCore::ShapeDetection::BarcodeDetector> RemoteBarcodeDetector::protectedBacking() const
{
    return backing();
}

Ref<RemoteRenderingBackend> RemoteBarcodeDetector::protectedBackend() const
{
    return m_backend.get();
}

void RemoteBarcodeDetector::detect(WebCore::RenderingResourceIdentifier renderingResourceIdentifier, CompletionHandler<void(Vector<WebCore::ShapeDetection::DetectedBarcode>&&)>&& completionHandler)
{
    auto sourceImage = protectedBackend()->imageBuffer(renderingResourceIdentifier);
    if (!sourceImage) {
        completionHandler({ });
        return;
    }

    protectedBacking()->detect(*sourceImage, WTFMove(completionHandler));
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
