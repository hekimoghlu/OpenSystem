/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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

#include "AnimationFrameRate.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

// Used to represent a given update. An value of { 3, 60 } indicates that this is the third update in a 1-second interval
// on a 60fps cadence. updateIndex will reset to zero every second, so { 59, 60 } is followed by { 0, 60 }.
struct DisplayUpdate {
    unsigned updateIndex { 0 };
    FramesPerSecond updatesPerSecond { 0 };
    
    DisplayUpdate nextUpdate() const
    {
        return { (updateIndex + 1) % updatesPerSecond, updatesPerSecond };
    }
    
    WEBCORE_EXPORT bool relevantForUpdateFrequency(FramesPerSecond) const;
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const DisplayUpdate&);

} // namespace WebCore
