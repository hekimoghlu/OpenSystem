/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#if ENABLE(MEDIA_SESSION)

namespace WebCore {

struct MediaPositionState {
    double duration = std::numeric_limits<double>::infinity();
    double playbackRate = 1;
    double position = 0;

    String toJSONString() const;

    friend bool operator==(const MediaPositionState&, const MediaPositionState&) = default;
};

}

namespace WTF {

template<typename> struct LogArgument;

template<> struct LogArgument<WebCore::MediaPositionState> {
    static String toString(const WebCore::MediaPositionState& state) { return state.toJSONString(); }
};

} // namespace WTF

#endif // ENABLE(MEDIA_SESSION)
