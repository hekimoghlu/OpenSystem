/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#include "CalculationRandomKey.h"
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {
namespace Calculation {

class RandomKeyMap final : public RefCounted<RandomKeyMap> {
public:
    static Ref<RandomKeyMap> create()
    {
        return adoptRef(*new RandomKeyMap);
    }

    double lookupUnitInterval(AtomString identifier, double min, double max, std::optional<double> step)
    {
        return m_map.ensure(RandomKey { identifier, min, max, step }, [] -> double {
            // FIXME: Probably doesn't need to be cryptographically strong, but starting with this.
            return cryptographicallyRandomUnitInterval();
        }).iterator->value;
    }

private:
    RandomKeyMap() = default;

    HashMap<RandomKey, double> m_map;
};

} // namespace Calculation
} // namespace WebCore
