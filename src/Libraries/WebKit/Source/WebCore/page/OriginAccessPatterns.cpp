/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include "OriginAccessPatterns.h"

#include "UserContentURLPattern.h"
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RuntimeApplicationChecks.h>

namespace WebCore {

// FIXME: Instead of having a singleton, this should be owned by Page.
OriginAccessPatternsForWebProcess& OriginAccessPatternsForWebProcess::singleton()
{
    // FIXME: We ought to be able to assert that isInWebProcess() is true, but
    // WebKitLegacy doesn't have a web content process and API::ContentRuleListStore
    // uses a CSSParserContext in the UI process. Use EmptyOriginAccessPatterns for that.
    ASSERT(!isInNetworkProcess());
    ASSERT(!isInGPUProcess());

    static NeverDestroyed<OriginAccessPatternsForWebProcess> instance;
    return instance.get();
}

static Lock originAccessPatternLock;
static Vector<UserContentURLPattern>& originAccessPatterns() WTF_REQUIRES_LOCK(originAccessPatternLock)
{
    ASSERT(originAccessPatternLock.isHeld());
    static NeverDestroyed<Vector<UserContentURLPattern>> originAccessPatterns;
    return originAccessPatterns;
}

void OriginAccessPatternsForWebProcess::allowAccessTo(const UserContentURLPattern& pattern)
{
    Locker locker { originAccessPatternLock };
    originAccessPatterns().append(pattern);
}

bool OriginAccessPatternsForWebProcess::anyPatternMatches(const URL& url) const
{
    Locker locker { originAccessPatternLock };
    for (const auto& pattern : originAccessPatterns()) {
        if (pattern.matches(url))
            return true;
    }
    return false;
}

const EmptyOriginAccessPatterns& EmptyOriginAccessPatterns::singleton()
{
    ASSERT(!isInWebProcess());
    static NeverDestroyed<EmptyOriginAccessPatterns> instance;
    return instance.get();
}

bool EmptyOriginAccessPatterns::anyPatternMatches(const URL&) const
{
    return false;
}

const OriginAccessPatterns& originAccessPatternsForWebProcessOrEmpty()
{
    if (isInWebProcess())
        return OriginAccessPatternsForWebProcess::singleton();
    return EmptyOriginAccessPatterns::singleton();
}

}
