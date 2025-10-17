/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#if ENABLE(GPU_PROCESS)
#include "PrepareBackingStoreBuffersData.h"

#include <wtf/text/TextStream.h>

namespace WebKit {

TextStream& operator<<(TextStream& ts, const ImageBufferSetPrepareBufferForDisplayInputData& inputData)
{
    ts << "remoteImageBufferSet: " << inputData.remoteBufferSet;
    ts << " dirtyRegion: " << inputData.dirtyRegion;
    ts << " supportsPartialRepaint: " << inputData.supportsPartialRepaint;
    ts << " hasEmptyDirtyRegion: " << inputData.hasEmptyDirtyRegion;
    ts << " requiresClearedPixels: " << inputData.requiresClearedPixels;
    return ts;
}

TextStream& operator<<(TextStream& ts, const ImageBufferSetPrepareBufferForDisplayOutputData& outputData)
{
    ts << "displayRequirement: " << outputData.displayRequirement;
    ts << "bufferCacheIdentifiers: " << outputData.bufferCacheIdentifiers;
    return ts;
}

TextStream& operator<<(TextStream& ts, SwapBuffersDisplayRequirement displayRequirement)
{
    switch (displayRequirement) {
    case SwapBuffersDisplayRequirement::NeedsFullDisplay: ts << "full display"; break;
    case SwapBuffersDisplayRequirement::NeedsNormalDisplay: ts << "normal display"; break;
    case SwapBuffersDisplayRequirement::NeedsNoDisplay: ts << "no display"; break;
    }

    return ts;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && PLATFORM(COCOA)
