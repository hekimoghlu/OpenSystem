/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

#if ENABLE(VIDEO)

#include <wtf/MediaTime.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>

namespace WebCore {

class MediaFragmentURIParser final {
public:
    
    MediaFragmentURIParser(const URL&);

    MediaTime startTime();
    MediaTime endTime();

private:

    void parseFragments();
    
    enum TimeFormat { None, Invalid, NormalPlayTime, SMPTETimeCode, WallClockTimeCode };
    void parseTimeFragment();
    bool parseNPTFragment(std::span<const LChar>, MediaTime& startTime, MediaTime& endTime);
    bool parseNPTTime(std::span<const LChar>, unsigned& offset, MediaTime&);

    URL m_url;
    TimeFormat m_timeFormat;
    MediaTime m_startTime;
    MediaTime m_endTime;
    Vector<std::pair<String, String>> m_fragments;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
