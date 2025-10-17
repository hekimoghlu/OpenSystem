/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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

#include "Length.h"
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

struct StyleMarqueeData : RefCounted<StyleMarqueeData> {
    static Ref<StyleMarqueeData> create();
    Ref<StyleMarqueeData> copy() const;

    bool operator==(const StyleMarqueeData&) const;

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleMarqueeData&) const;
#endif

    Length increment;
    int speed;
    int loops; // -1 means infinite.
    unsigned behavior : 2; // MarqueeBehavior 
    unsigned direction : 3; // MarqueeDirection

private:
    StyleMarqueeData();
    StyleMarqueeData(const StyleMarqueeData&);
};

} // namespace WebCore
