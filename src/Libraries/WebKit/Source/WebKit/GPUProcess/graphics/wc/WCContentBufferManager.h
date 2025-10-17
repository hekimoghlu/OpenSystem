/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

#if USE(GRAPHICS_LAYER_WC)

#include "WCContentBufferIdentifier.h"
#include <WebCore/PlatformLayer.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/HashMap.h>

namespace WebCore {
class TextureMapperPlatformLayer;
}

namespace WebKit {

class WCContentBuffer;

class WCContentBufferManager {
public:
    class ProcessInfo;

    static WCContentBufferManager& singleton();

    std::optional<WCContentBufferIdentifier> acquireContentBufferIdentifier(WebCore::ProcessIdentifier, WebCore::TextureMapperPlatformLayer*);
    WCContentBuffer* releaseContentBufferIdentifier(WebCore::ProcessIdentifier, WCContentBufferIdentifier);
    void removeContentBuffer(WebCore::ProcessIdentifier, WCContentBuffer&);
    void removeAllContentBuffersForProcess(WebCore::ProcessIdentifier);

private:
    HashMap<WebCore::ProcessIdentifier, std::unique_ptr<ProcessInfo>> m_processMap;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
