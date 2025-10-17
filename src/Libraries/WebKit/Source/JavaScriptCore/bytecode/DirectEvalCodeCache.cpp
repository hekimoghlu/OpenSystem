/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "DirectEvalCodeCache.h"

#include "JSCellInlines.h"

namespace JSC {

void DirectEvalCodeCache::setSlow(JSGlobalObject* globalObject, JSCell* owner, const String& evalSource, BytecodeIndex bytecodeIndex, DirectEvalExecutable* evalExecutable)
{
    if (!evalExecutable->allowDirectEvalCache())
        return;

    Locker locker { m_lock };
    m_cacheMap.set(CacheKey(evalSource, bytecodeIndex), WriteBarrier<DirectEvalExecutable>(globalObject->vm(), owner, evalExecutable));
}

void DirectEvalCodeCache::clear()
{
    Locker locker { m_lock };
    m_cacheMap.clear();
}

template<typename Visitor>
void DirectEvalCodeCache::visitAggregateImpl(Visitor& visitor)
{
    Locker locker { m_lock };
    EvalCacheMap::iterator end = m_cacheMap.end();
    for (EvalCacheMap::iterator ptr = m_cacheMap.begin(); ptr != end; ++ptr)
        visitor.append(ptr->value);
}

DEFINE_VISIT_AGGREGATE(DirectEvalCodeCache);

} // namespace JSC
