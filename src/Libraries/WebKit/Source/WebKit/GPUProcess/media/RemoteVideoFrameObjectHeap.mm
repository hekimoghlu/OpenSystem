/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include "RemoteVideoFrameObjectHeap.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO) && PLATFORM(COCOA)

#include <WebCore/PixelBufferConformerCV.h>
#include <WebCore/CoreVideoSoftLink.h>

namespace WebKit {

RefPtr<WebCore::VideoFrame> RemoteVideoFrameObjectHeap::get(RemoteVideoFrameReadReference&& read)
{
    return m_heap.read(WTFMove(read), 0_s);
}

UniqueRef<WebCore::PixelBufferConformerCV> RemoteVideoFrameObjectHeap::createPixelConformer()
{
    return makeUniqueRef<WebCore::PixelBufferConformerCV>((__bridge CFDictionaryRef)@{ (__bridge NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA) });
}

}

#endif
