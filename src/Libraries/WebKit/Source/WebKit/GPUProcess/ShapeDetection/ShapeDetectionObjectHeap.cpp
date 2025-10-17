/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include "ShapeDetectionObjectHeap.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteBarcodeDetector.h"
#include "RemoteFaceDetector.h"
#include "RemoteRenderingBackend.h"
#include "RemoteTextDetector.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::ShapeDetection {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ObjectHeap);

ObjectHeap::ObjectHeap() = default;

ObjectHeap::~ObjectHeap() = default;

void ObjectHeap::addObject(ShapeDetectionIdentifier identifier, RemoteBarcodeDetector& barcodeDetector)
{
    auto result = m_barcodeDetectors.add(identifier, barcodeDetector);
    ASSERT_UNUSED(result, result.isNewEntry);
}

void ObjectHeap::addObject(ShapeDetectionIdentifier identifier, RemoteFaceDetector& faceDetector)
{
    auto result = m_faceDetectors.add(identifier, faceDetector);
    ASSERT_UNUSED(result, result.isNewEntry);
}

void ObjectHeap::addObject(ShapeDetectionIdentifier identifier, RemoteTextDetector& textDetector)
{
    auto result = m_textDetectors.add(identifier, textDetector);
    ASSERT_UNUSED(result, result.isNewEntry);
}

void ObjectHeap::removeObject(ShapeDetectionIdentifier identifier)
{
    int count = 0;
    count += m_barcodeDetectors.remove(identifier);
    count += m_faceDetectors.remove(identifier);
    count += m_textDetectors.remove(identifier);
    ASSERT_UNUSED(count, count == 1);
}

void ObjectHeap::clear()
{
    m_barcodeDetectors.clear();
    m_faceDetectors.clear();
    m_textDetectors.clear();
}

} // namespace WebKit::ShapeDetection

#endif // ENABLE(GPU_PROCESS)
