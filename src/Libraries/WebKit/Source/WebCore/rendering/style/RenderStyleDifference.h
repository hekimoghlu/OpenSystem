/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#if !LOG_DISABLED

#include <wtf/PointerComparison.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

#define LOG_IF_DIFFERENT(name) \
    do { logIfDifferent(ts, ASCIILiteral::fromLiteralUnsafe(#name), name, other.name); } while (0)

#define LOG_IF_DIFFERENT_WITH_CAST(type, name) \
    do { logIfDifferent(ts, ASCIILiteral::fromLiteralUnsafe(#name), static_cast<type>(name), static_cast<type>(other.name)); } while (0)

#define LOG_RAW_OPTIONSET_IF_DIFFERENT(type, name) \
    do { logIfDifferent(ts, ASCIILiteral::fromLiteralUnsafe(#name), OptionSet<type>::fromRaw(name), OptionSet<type>::fromRaw(other.name)); } while (0)


template<class T>
struct is_pointer_wrapper : std::false_type { };

template<class T>
struct is_pointer_wrapper<RefPtr<T>> : std::true_type { };

template<class T>
struct is_pointer_wrapper<Ref<T>> : std::true_type { };

template<class T>
struct is_pointer_wrapper<std::unique_ptr<T>> : std::true_type { };

template<typename T>
struct ValueOrUnstreamableMessage {
    explicit ValueOrUnstreamableMessage(const T& value)
        : value(value)
    { }
    const T& value;
};

template<typename T>
TextStream& operator<<(TextStream& ts, ValueOrUnstreamableMessage<T> item)
{
    if constexpr (WTF::supports_text_stream_insertion<T>::value) {
        if constexpr (is_pointer_wrapper<T>::value)
            ts << ValueOrNull(item.value.get());
        else
            ts << item.value;
    } else
        ts << "(unstreamable)";
    return ts;
}


template<typename T>
void logIfDifferent(TextStream& ts, ASCIILiteral name, const T& item1, const T& item2)
{
    bool differ = false;
    if constexpr (is_pointer_wrapper<T>::value)
        differ = !arePointingToEqualData(item1, item2);
    else
        differ = item1 != item2;

    if (differ)
        ts << name << " differs: " << ValueOrUnstreamableMessage(item1) << ", " << ValueOrUnstreamableMessage(item2) << '\n';
}

} // namespace WebCore

#endif // !LOG_DISABLED
