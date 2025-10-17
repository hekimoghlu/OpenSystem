/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

namespace WebCore {

struct DateTimeFieldsState {
    enum class Meridiem : bool { AM, PM };

    std::optional<unsigned> hour23() const
    {
        if (!hour || !meridiem)
            return std::nullopt;
        return (*hour % 12) + (*meridiem == DateTimeFieldsState::Meridiem::PM ? 12 : 0);
    }

    std::optional<unsigned> year;
    std::optional<unsigned> month;
    std::optional<unsigned> week;
    std::optional<unsigned> dayOfMonth;
    std::optional<unsigned> hour;
    std::optional<unsigned> minute;
    std::optional<unsigned> second;
    std::optional<unsigned> millisecond;
    std::optional<Meridiem> meridiem;
};

} // namespace WebCore
