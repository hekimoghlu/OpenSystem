/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
#include "ChromeClient.h"

#include "BarcodeDetectorInterface.h"
#include "BarcodeDetectorOptionsInterface.h"
#include "BarcodeFormatInterface.h"
#include "FaceDetectorInterface.h"
#include "FaceDetectorOptionsInterface.h"
#include "ScrollbarsController.h"
#include "ScrollingCoordinator.h"
#include "TextDetectorInterface.h"
#include "WorkerClient.h"

#if ENABLE(WEBGL)
#include "GraphicsContextGL.h"
#endif

namespace WebCore {

ChromeClient::ChromeClient() = default;

ChromeClient::~ChromeClient() = default;

std::unique_ptr<WorkerClient> ChromeClient::createWorkerClient(SerialFunctionDispatcher&)
{
    return nullptr;
}

#if ENABLE(WEBGL)
RefPtr<GraphicsContextGL> ChromeClient::createGraphicsContextGL(const GraphicsContextGLAttributes& attributes) const
{
    return createWebProcessGraphicsContextGL(attributes);
}
#endif

RefPtr<ImageBuffer> ChromeClient::sinkIntoImageBuffer(std::unique_ptr<WebCore::SerializedImageBuffer> imageBuffer)
{
    return SerializedImageBuffer::sinkIntoImageBuffer(WTFMove(imageBuffer));
}

void ChromeClient::ensureScrollbarsController(Page&, ScrollableArea& area, bool update) const
{
    if (update)
        return;

    area.ScrollableArea::createScrollbarsController();
}

RefPtr<ScrollingCoordinator> ChromeClient::createScrollingCoordinator(Page&) const
{
    return nullptr;
}

RefPtr<ShapeDetection::BarcodeDetector> ChromeClient::createBarcodeDetector(const ShapeDetection::BarcodeDetectorOptions&) const
{
    return nullptr;
}

void ChromeClient::getBarcodeDetectorSupportedFormats(CompletionHandler<void(Vector<ShapeDetection::BarcodeFormat>&&)>&& completionHandler) const
{
    completionHandler({ });
}

RefPtr<ShapeDetection::FaceDetector> ChromeClient::createFaceDetector(const ShapeDetection::FaceDetectorOptions&) const
{
    return nullptr;
}

RefPtr<ShapeDetection::TextDetector> ChromeClient::createTextDetector() const
{
    return nullptr;
}

} // namespace WebCore
