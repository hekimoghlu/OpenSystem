/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#include <wtf/StdLibExtras.h>

#if USE(SKIA)

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkPixmap.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

inline std::span<const uint8_t> span(SkData* data)
{
    return unsafeMakeSpan<const uint8_t>(data->bytes(), data->size());
}

inline std::span<const uint8_t> span(const sk_sp<SkData>& data)
{
    return span(data.get());
}

inline std::span<const uint8_t> span(const SkPixmap& pixmap)
{
    return unsafeMakeSpan(static_cast<const uint8_t*>(pixmap.addr()), pixmap.computeByteSize());
}

inline std::span<uint8_t> mutableSpan(SkPixmap& pixmap)
{
    return unsafeMakeSpan(static_cast<uint8_t*>(pixmap.writable_addr()), pixmap.computeByteSize());
}

} // namespace WebCore

using WebCore::span;

#endif // USE(SKIA)
