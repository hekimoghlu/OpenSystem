/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#include <wtf/WallTime.h>

namespace WebCore::PCM {

enum class AttributionReportEndpoint : bool { Source, Destination };

struct AttributionTimeToSendData {
    std::optional<WallTime> sourceEarliestTimeToSend;
    std::optional<WallTime> destinationEarliestTimeToSend;

    std::optional<WallTime> earliestTimeToSend()
    {
        if (!sourceEarliestTimeToSend && !destinationEarliestTimeToSend)
            return std::nullopt;

        if (sourceEarliestTimeToSend && destinationEarliestTimeToSend)
            return std::min(sourceEarliestTimeToSend, destinationEarliestTimeToSend);

        return sourceEarliestTimeToSend ? sourceEarliestTimeToSend : destinationEarliestTimeToSend;
    }

    std::optional<WallTime> latestTimeToSend()
    {
        if (!sourceEarliestTimeToSend && !destinationEarliestTimeToSend)
            return std::nullopt;

        if (sourceEarliestTimeToSend && destinationEarliestTimeToSend)
            return std::max(sourceEarliestTimeToSend, destinationEarliestTimeToSend);

        return sourceEarliestTimeToSend ? sourceEarliestTimeToSend : destinationEarliestTimeToSend;
    }

    std::optional<AttributionReportEndpoint> attributionReportEndpoint()
    {
        if (sourceEarliestTimeToSend && destinationEarliestTimeToSend) {
            if (*sourceEarliestTimeToSend < *destinationEarliestTimeToSend)
                return AttributionReportEndpoint::Source;

            return AttributionReportEndpoint::Destination;
        }

        if (sourceEarliestTimeToSend)
            return AttributionReportEndpoint::Source;

        if (destinationEarliestTimeToSend)
            return AttributionReportEndpoint::Destination;

        return std::nullopt;
    }
};

}
