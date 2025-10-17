/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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

#if ENABLE(GPU_PROCESS)

#include "BufferIdentifierSet.h"
#include "ImageBufferBackendHandle.h"
#include "RemoteImageBufferSetIdentifier.h"
#include "SwapBuffersDisplayRequirement.h"
#include <WebCore/Region.h>
#include <WebCore/RenderingResourceIdentifier.h>

namespace WTF {
class TextStream;
}

namespace WebKit {

struct ImageBufferSetPrepareBufferForDisplayInputData {
    RemoteImageBufferSetIdentifier remoteBufferSet;
    WebCore::Region dirtyRegion;
    bool supportsPartialRepaint { true };
    bool hasEmptyDirtyRegion { true };
    bool requiresClearedPixels { true };
};

struct ImageBufferSetPrepareBufferForDisplayOutputData {
    std::optional<ImageBufferBackendHandle> backendHandle;
    SwapBuffersDisplayRequirement displayRequirement { SwapBuffersDisplayRequirement::NeedsNoDisplay };
    BufferIdentifierSet bufferCacheIdentifiers;
};

WTF::TextStream& operator<<(WTF::TextStream&, const ImageBufferSetPrepareBufferForDisplayInputData&);
WTF::TextStream& operator<<(WTF::TextStream&, const ImageBufferSetPrepareBufferForDisplayOutputData&);

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
