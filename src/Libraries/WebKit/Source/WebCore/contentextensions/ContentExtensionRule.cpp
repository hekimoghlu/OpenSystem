/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "ContentExtensionRule.h"

#include <wtf/CrossThreadCopier.h>

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore::ContentExtensions {

ContentExtensionRule::ContentExtensionRule(Trigger&& trigger, Action&& action)
    : m_trigger(WTFMove(trigger))
    , m_action(WTFMove(action))
{
    ASSERT(!m_trigger.urlFilter.isEmpty());
}

template<size_t index, typename... Types>
struct VariantDeserializerHelper {
    using VariantType = typename std::variant_alternative<index, std::variant<Types...>>::type;
    static std::variant<Types...> deserialize(std::span<const uint8_t> span, size_t i)
    {
        if (i == index)
            return VariantType::deserialize(span);
        return VariantDeserializerHelper<index - 1, Types...>::deserialize(span, i);
    }
    static size_t serializedLength(std::span<const uint8_t> span, size_t i)
    {
        if (i == index)
            return VariantType::serializedLength(span);
        return VariantDeserializerHelper<index - 1, Types...>::serializedLength(span, i);
    }
};

template<typename... Types>
struct VariantDeserializerHelper<0, Types...> {
    using VariantType = typename std::variant_alternative<0, std::variant<Types...>>::type;
    static std::variant<Types...> deserialize(std::span<const uint8_t> span, size_t i)
    {
        ASSERT_UNUSED(i, !i);
        return VariantType::deserialize(span);
    }
    static size_t serializedLength(std::span<const uint8_t> span, size_t i)
    {
        ASSERT_UNUSED(i, !i);
        return VariantType::serializedLength(span);
    }
};

template<typename T> struct VariantDeserializer;
template<typename... Types> struct VariantDeserializer<std::variant<Types...>> {
    static std::variant<Types...> deserialize(std::span<const uint8_t> span, size_t i)
    {
        return VariantDeserializerHelper<sizeof...(Types) - 1, Types...>::deserialize(span, i);
    }
    static size_t serializedLength(std::span<const uint8_t> span, size_t i)
    {
        return VariantDeserializerHelper<sizeof...(Types) - 1, Types...>::serializedLength(span, i);
    }
};

DeserializedAction DeserializedAction::deserialize(std::span<const uint8_t> serializedActions, uint32_t location)
{
    RELEASE_ASSERT(location < serializedActions.size());
    return { location, VariantDeserializer<ActionData>::deserialize(serializedActions.subspan(location + 1), serializedActions[location]) };
}

size_t DeserializedAction::serializedLength(std::span<const uint8_t> serializedActions, uint32_t location)
{
    RELEASE_ASSERT(location < serializedActions.size());
    return 1 + VariantDeserializer<ActionData>::serializedLength(serializedActions.subspan(location + 1), serializedActions[location]);
}

Trigger Trigger::isolatedCopy() const &
{
    return { urlFilter.isolatedCopy(), urlFilterIsCaseSensitive, topURLFilterIsCaseSensitive, frameURLFilterIsCaseSensitive, flags, crossThreadCopy(conditions) };
}

Trigger Trigger::isolatedCopy() &&
{
    return { WTFMove(urlFilter).isolatedCopy(), urlFilterIsCaseSensitive, topURLFilterIsCaseSensitive, frameURLFilterIsCaseSensitive, flags, crossThreadCopy(WTFMove(conditions)) };
}

Action Action::isolatedCopy() const &
{
    return { crossThreadCopy(m_data) };
}

Action Action::isolatedCopy() &&
{
    return { crossThreadCopy(WTFMove(m_data)) };
}

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
