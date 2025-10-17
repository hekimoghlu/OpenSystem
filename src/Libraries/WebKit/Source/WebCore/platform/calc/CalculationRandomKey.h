/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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

#include <optional>
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/text/AtomString.h>

namespace WebCore {
namespace Calculation {

struct RandomKey {
    AtomString identifier;
    double min;
    double max;
    std::optional<double> step;

    RandomKey(AtomString identifier, double min, double max, std::optional<double> step)
        : identifier { WTFMove(identifier) }
        , min { min }
        , max { max }
        , step { step }
    {
        RELEASE_ASSERT(!std::isnan(min));
        RELEASE_ASSERT(!std::isnan(max));
    }

    explicit RandomKey(WTF::HashTableDeletedValueType)
        : identifier { }
        , min { std::numeric_limits<double>::quiet_NaN() }
        , max { 0 }
        , step { std::nullopt }
    {
    }

    explicit RandomKey(WTF::HashTableEmptyValueType)
        : identifier { }
        , min { 0 }
        , max { std::numeric_limits<double>::quiet_NaN() }
        , step { std::nullopt }
    {
    }

    bool isHashTableDeletedValue() const { return std::isnan(min); }
    bool isHashTableEmptyValue() const { return std::isnan(max); }

    bool operator==(const RandomKey&) const = default;
};

} // namespace Calculation
} // namespace WebCore

namespace WTF {

struct CalculationRandomKeyHash {
    static unsigned hash(const WebCore::Calculation::RandomKey& key) { return computeHash(key.identifier, key.min, key.max, key.step); }
    static bool equal(const WebCore::Calculation::RandomKey& a, const WebCore::Calculation::RandomKey& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<> struct HashTraits<WebCore::Calculation::RandomKey> : GenericHashTraits<WebCore::Calculation::RandomKey> {
    static WebCore::Calculation::RandomKey emptyValue() { return WebCore::Calculation::RandomKey(HashTableEmptyValue); }
    static bool isEmptyValue(const WebCore::Calculation::RandomKey& value) { return value.isHashTableEmptyValue(); }
    static void constructDeletedValue(WebCore::Calculation::RandomKey& slot) { new (NotNull, &slot) WebCore::Calculation::RandomKey(HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::Calculation::RandomKey& slot) { return slot.isHashTableDeletedValue(); }

    static const bool hasIsEmptyValueFunction = true;
    static const bool emptyValueIsZero = false;
};

template<> struct DefaultHash<WebCore::Calculation::RandomKey> : CalculationRandomKeyHash { };

} // namespace WTF
