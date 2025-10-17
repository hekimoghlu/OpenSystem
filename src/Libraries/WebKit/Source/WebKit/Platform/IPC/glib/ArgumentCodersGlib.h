/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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

#include <glib.h>
#include <optional>

#include <wtf/glib/GSpanExtras.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

namespace IPC {

template<> struct ArgumentCoder<GUniquePtr<char*>> {
    static void encode(Encoder& encoder, const GUniquePtr<char*>& strv)
    {
        auto length = strv.get() ? g_strv_length(strv.get()) : 0;

        encoder << length;

        if (!length)
            return;

        auto strvSpan = span(strv.get());
        for (auto str : strvSpan)
            encoder << CString(str);
    }

    static std::optional<GUniquePtr<char*>> decode(Decoder& decoder)
    {
        auto length = decoder.decode<unsigned>();

        if (UNLIKELY(!length))
            return std::nullopt;

        GUniquePtr<char*>strv(g_new0(char*, *length + 1));
        auto strvSpan = unsafeMakeSpan(strv.get(), *length);

        for (auto& str : strvSpan) {
            auto strOptional = decoder.decode<CString>();
            if (UNLIKELY(!strOptional))
                return std::nullopt;
            str = g_strdup(strOptional->data());
        }

        return strv;
    }
};

} // namespace IPC
