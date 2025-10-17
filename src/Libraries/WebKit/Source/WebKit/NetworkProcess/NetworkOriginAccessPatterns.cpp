/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
#include "config.h"
#include "NetworkOriginAccessPatterns.h"

#include <wtf/Algorithms.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkOriginAccessPatterns);

void NetworkOriginAccessPatterns::allowAccessTo(const WebCore::UserContentURLPattern& pattern)
{
    ASSERT(RunLoop::isMain());
    m_patterns.append(pattern);
}

bool NetworkOriginAccessPatterns::anyPatternMatches(const URL& url) const
{
    ASSERT(RunLoop::isMain());
    return WTF::anyOf(m_patterns, [&] (auto& pattern) {
        return pattern.matches(url);
    });
}

}
