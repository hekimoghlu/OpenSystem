/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

#include "WPEEvent.h"
#include <optional>

namespace WPE {

class GestureDetector final {
public:
    void handleEvent(WPEEvent*);
    WPEGesture gesture() const { return m_gesture; }
    void reset();

    struct Position {
        double x;
        double y;
    };
    using Delta = Position;

    std::optional<Position> position() const { return m_position; }
    std::optional<Delta> delta() const { return m_delta; }
    bool dragBegin() const { return m_dragBegin && *m_dragBegin; }

private:
    WPEGesture m_gesture { WPE_GESTURE_NONE };
    std::optional<uint32_t> m_sequenceId;
    std::optional<Position> m_position;
    std::optional<Position> m_nextDeltaReferencePosition;
    std::optional<Delta> m_delta;
    std::optional<bool> m_dragBegin;
};

} // namespace WPE
