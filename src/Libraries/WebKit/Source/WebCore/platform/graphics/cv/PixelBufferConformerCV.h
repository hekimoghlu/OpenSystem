/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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

#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

typedef struct CGColorSpace *CGColorSpaceRef;
typedef struct CGImage* CGImageRef;
typedef struct OpaqueVTPixelBufferConformer* VTPixelBufferConformerRef;
typedef struct __CVBuffer *CVPixelBufferRef;

namespace WebCore {

class PixelBufferConformerCV {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PixelBufferConformerCV, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT PixelBufferConformerCV(CFDictionaryRef attributes);
    WEBCORE_EXPORT RetainPtr<CVPixelBufferRef> convert(CVPixelBufferRef);
    WEBCORE_EXPORT RetainPtr<CGImageRef> createImageFromPixelBuffer(CVPixelBufferRef);
    static WEBCORE_EXPORT RetainPtr<CGImageRef> imageFrom32BGRAPixelBuffer(RetainPtr<CVPixelBufferRef>&&, CGColorSpaceRef);

private:
    RetainPtr<VTPixelBufferConformerRef> m_pixelConformer;
};

}
