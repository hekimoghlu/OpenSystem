/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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

#if USE(SKIA)

#include "ViewSnapshotStore.h"
#include <WebCore/NotImplemented.h>
#include <skia/core/SkData.h>
IGNORE_CLANG_WARNINGS_BEGIN("cast-align")
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Skia port
#include <skia/encode/SkPngEncoder.h>
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
IGNORE_CLANG_WARNINGS_END
#include <span>
#include <wtf/text/Base64.h>

namespace WebKit {
using namespace WebCore;

static std::optional<String> base64EncodedPNGData(SkImage& image)
{
    auto data = SkPngEncoder::Encode(nullptr, &image, { });
    if (!data)
        return std::nullopt;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Skia port
    return base64EncodeToString(std::span<const uint8_t>(data->bytes(), data->size()));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

std::optional<String> WebAutomationSession::platformGetBase64EncodedPNGData(ShareableBitmap::Handle&& handle)
{
    auto bitmap = ShareableBitmap::create(WTFMove(handle), SharedMemory::Protection::ReadOnly);
    if (!bitmap)
        return std::nullopt;

    auto image = bitmap->createPlatformImage();
    return base64EncodedPNGData(*image.get());
}

#if !PLATFORM(GTK)
std::optional<String> WebAutomationSession::platformGetBase64EncodedPNGData(const ViewSnapshot&)
{
    notImplemented();
    return std::nullopt;
}
#endif

} // namespace WebKit

#endif // USE(SKIA)
