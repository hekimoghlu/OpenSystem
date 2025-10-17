/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#include "WebAutomationSession.h"

#if USE(CAIRO)

#include "ViewSnapshotStore.h"
#include <WebCore/NotImplemented.h>
#include <WebCore/RefPtrCairo.h>
#include <cairo.h>
#include <wtf/text/Base64.h>

namespace WebKit {
using namespace WebCore;

static std::optional<String> base64EncodedPNGData(cairo_surface_t* surface)
{
    if (!surface)
        return std::nullopt;

    Vector<uint8_t> pngData;
    cairo_surface_write_to_png_stream(surface, [](void* userData, const unsigned char* data, unsigned length) -> cairo_status_t {
        auto* pngData = static_cast<Vector<uint8_t>*>(userData);
        pngData->append(std::span { reinterpret_cast<const uint8_t*>(data), length });
        return CAIRO_STATUS_SUCCESS;
    }, &pngData);

    if (pngData.isEmpty())
        return std::nullopt;

    return base64EncodeToString(pngData);
}

std::optional<String> WebAutomationSession::platformGetBase64EncodedPNGData(ShareableBitmap::Handle&& handle)
{
    auto bitmap = ShareableBitmap::create(WTFMove(handle), SharedMemory::Protection::ReadOnly);
    if (!bitmap)
        return std::nullopt;

    auto surface = bitmap->createCairoSurface();
    return base64EncodedPNGData(surface.get());
}

#if !PLATFORM(GTK)
std::optional<String> WebAutomationSession::platformGetBase64EncodedPNGData(const ViewSnapshot&)
{
    notImplemented();
    return std::nullopt;
}
#endif

} // namespace WebKit

#endif // USE(CAIRO)
