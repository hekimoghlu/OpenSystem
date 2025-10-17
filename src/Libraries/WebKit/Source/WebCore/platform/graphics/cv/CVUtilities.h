/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#include <CoreVideo/CoreVideo.h>
#include <utility>
#include <wtf/Expected.h>
#include <wtf/RetainPtr.h>

using CGColorSpaceRef = struct CGColorSpace*;
using CVPixelBufferPoolRef = struct __CVPixelBufferPool*;
using CVPixelBufferRef = struct __CVBuffer*;

namespace WebCore {
class ProcessIdentity;

// Creates CVPixelBufferPool that creates CVPixelBuffers backed by IOSurfaces.
// These buffers can be for example sent through IPC.
WEBCORE_EXPORT Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createIOSurfaceCVPixelBufferPool(size_t width, size_t height, OSType pixelFormat, unsigned minimumBufferCount = 0u, bool isCGImageCompatible = false);

WEBCORE_EXPORT Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createInMemoryCVPixelBufferPool(size_t width, size_t height, OSType pixelFormat, unsigned minimumBufferCount = 0u, bool isCGImageCompatible = false);

WEBCORE_EXPORT Expected<RetainPtr<CVPixelBufferPoolRef>, CVReturn> createCVPixelBufferPool(size_t width, size_t height, OSType pixelFormat, unsigned minimumBufferCount = 0u, bool isCGImageCompatible = false, bool shouldUseIOSUrface = true);

WEBCORE_EXPORT Expected<RetainPtr<CVPixelBufferRef>, CVReturn> createCVPixelBufferFromPool(CVPixelBufferPoolRef, unsigned maximumBufferCount = 0u);

WEBCORE_EXPORT Expected<RetainPtr<CVPixelBufferRef>, CVReturn> createCVPixelBuffer(IOSurfaceRef);

WEBCORE_EXPORT RetainPtr<CGColorSpaceRef> createCGColorSpaceForCVPixelBuffer(CVPixelBufferRef);

// Should be called with non-empty ProcessIdentity.
WEBCORE_EXPORT void setOwnershipIdentityForCVPixelBuffer(CVPixelBufferRef, const ProcessIdentity&);

WEBCORE_EXPORT RetainPtr<CVPixelBufferRef> createBlackPixelBuffer(size_t width, size_t height, bool shouldUseIOSurface = false);

}
