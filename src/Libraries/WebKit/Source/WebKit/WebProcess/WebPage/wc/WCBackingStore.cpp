/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#include "WCBackingStore.h"

#if USE(GRAPHICS_LAYER_WC)

#include "ImageBufferBackendHandleSharing.h"
#include <WebCore/ShareableBitmap.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WCBackingStore);

WCBackingStore::WCBackingStore(std::optional<ImageBufferBackendHandle>&& handle)
{
    if (auto* imageHandle = handle ? std::get_if<WebCore::ShareableBitmap::Handle>(&*handle) : nullptr)
        m_bitmap = WebCore::ShareableBitmap::create(WTFMove(*imageHandle));
}

std::optional<ImageBufferBackendHandle> WCBackingStore::handle() const
{
    if (!m_imageBuffer)
        return std::nullopt;

    auto* sharing = m_imageBuffer->toBackendSharing();
    if (!is<ImageBufferBackendHandleSharing>(sharing))
        return std::nullopt;

    return dynamicDowncast<ImageBufferBackendHandleSharing>(*sharing)->createBackendHandle();
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
