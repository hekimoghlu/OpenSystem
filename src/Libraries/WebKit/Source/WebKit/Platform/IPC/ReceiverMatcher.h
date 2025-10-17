/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

namespace IPC {

struct ReceiverMatcher {
    // Matches all messages.
    ReceiverMatcher() = default;

    // Matches message to specific receiver, any destination ID.
    ReceiverMatcher(ReceiverName receiverName)
        : receiverName(receiverName)
    {
    }

    // Matches message to specific receiver, specific destination ID.
    // Note: destinationID == 0 matches only 0 ids.
    ReceiverMatcher(ReceiverName receiverName, uint64_t destinationID)
        : receiverName(receiverName)
        , destinationID(destinationID)
    {
    }

    // Creates a matcher from parameters where destinationID == 0 means any destintation ID. Deprecated.
    static ReceiverMatcher createWithZeroAsAnyDestination(ReceiverName receiverName, uint64_t destinationID)
    {
        if (destinationID)
            return ReceiverMatcher { receiverName, destinationID };
        return ReceiverMatcher { receiverName };
    }

    bool matches(ReceiverName matchReceiverName, uint64_t matchDestinationID) const
    {
        return !receiverName || (*receiverName == matchReceiverName && (!destinationID || *destinationID == matchDestinationID));
    }

    std::optional<ReceiverName> receiverName;
    std::optional<uint64_t> destinationID;
};

}
