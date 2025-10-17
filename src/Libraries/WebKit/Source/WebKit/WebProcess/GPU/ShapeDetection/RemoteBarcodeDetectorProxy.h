/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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

#include "RenderingBackendIdentifier.h"
#include "ShapeDetectionIdentifier.h"
#include <WebCore/BarcodeDetectorInterface.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if ENABLE(GPU_PROCESS)

namespace IPC {
class StreamClientConnection;
}

namespace WebCore::ShapeDetection {
struct BarcodeDetectorOptions;
enum class BarcodeFormat : uint8_t;
struct DetectedBarcode;
}

namespace WebKit::ShapeDetection {

class RemoteBarcodeDetectorProxy : public WebCore::ShapeDetection::BarcodeDetector {
    WTF_MAKE_TZONE_ALLOCATED(RemoteBarcodeDetectorProxy);
public:
    static Ref<RemoteBarcodeDetectorProxy> create(Ref<IPC::StreamClientConnection>&&, RenderingBackendIdentifier, ShapeDetectionIdentifier, const WebCore::ShapeDetection::BarcodeDetectorOptions&);

    virtual ~RemoteBarcodeDetectorProxy();

    static void getSupportedFormats(Ref<IPC::StreamClientConnection>&&, RenderingBackendIdentifier, CompletionHandler<void(Vector<WebCore::ShapeDetection::BarcodeFormat>&&)>&&);

private:
    RemoteBarcodeDetectorProxy(Ref<IPC::StreamClientConnection>&&, RenderingBackendIdentifier, ShapeDetectionIdentifier);

    RemoteBarcodeDetectorProxy(const RemoteBarcodeDetectorProxy&) = delete;
    RemoteBarcodeDetectorProxy(RemoteBarcodeDetectorProxy&&) = delete;
    RemoteBarcodeDetectorProxy& operator=(const RemoteBarcodeDetectorProxy&) = delete;
    RemoteBarcodeDetectorProxy& operator=(RemoteBarcodeDetectorProxy&&) = delete;

    ShapeDetectionIdentifier backing() const { return m_backing; }

    void detect(Ref<WebCore::ImageBuffer>&&, CompletionHandler<void(Vector<WebCore::ShapeDetection::DetectedBarcode>&&)>&&) final;

    ShapeDetectionIdentifier m_backing;
    Ref<IPC::StreamClientConnection> m_streamClientConnection;
    RenderingBackendIdentifier m_renderingBackendIdentifier;
};

} // namespace WebKit::ShapeDetection

#endif // ENABLE(GPU_PROCESS)
