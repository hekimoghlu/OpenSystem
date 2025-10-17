/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#include "LazyValueProfile.h"

#include "JSCJSValueInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_STRUCT_TZONE_ALLOCATED_IMPL(CompressedLazyValueProfileHolder::LazyValueProfileHolder);

void CompressedLazyValueProfileHolder::computeUpdatedPredictions(const ConcurrentJSLocker& locker, CodeBlock* codeBlock)
{
    if (!m_data)
        return;

    for (auto& profile : m_data->operandValueProfiles)
        profile.computeUpdatedPrediction(locker);

    for (auto& pair : m_data->speculationFailureValueProfileBuckets) {
        ValueProfile& profile = codeBlock->valueProfileForBytecodeIndex(pair.first);
        profile.computeUpdatedPredictionForExtraValue(locker, pair.second);
    }
}

void CompressedLazyValueProfileHolder::initializeData()
{
    ASSERT(!isCompilationThread());
    ASSERT(!m_data);
    auto data = makeUnique<LazyValueProfileHolder>();
    // Make sure the initialization of the holder happens before we expose the data to compiler threads.
    WTF::storeStoreFence();
    m_data = WTFMove(data);
}

LazyOperandValueProfile* CompressedLazyValueProfileHolder::addOperandValueProfile(const LazyOperandValueProfileKey& key)
{
    // This addition happens from mutator thread. Thus, we do not need to consider about concurrent additions from multiple threads.
    ASSERT(!isCompilationThread());

    if (!m_data)
        initializeData();

    for (auto& profile : m_data->operandValueProfiles) {
        if (profile.key() == key)
            return &profile;
    }

    m_data->operandValueProfiles.appendConcurrently(LazyOperandValueProfile(key));
    return &m_data->operandValueProfiles.last();
}

JSValue* CompressedLazyValueProfileHolder::addSpeculationFailureValueProfile(BytecodeIndex index)
{
    // This addition happens from mutator thread. Thus, we do not need to consider about concurrent additions from multiple threads.
    ASSERT(!isCompilationThread());

    if (!m_data)
        initializeData();

    for (auto& pair : m_data->speculationFailureValueProfileBuckets) {
        if (pair.first == index)
            return &pair.second;
    }

    m_data->speculationFailureValueProfileBuckets.appendConcurrently(std::make_pair(index, JSValue()));
    return &m_data->speculationFailureValueProfileBuckets.last().second;
}

UncheckedKeyHashMap<BytecodeIndex, JSValue*> CompressedLazyValueProfileHolder::speculationFailureValueProfileBucketsMap()
{
    UncheckedKeyHashMap<BytecodeIndex, JSValue*> result;
    if (m_data) {
        result.reserveInitialCapacity(m_data->speculationFailureValueProfileBuckets.size());
        for (auto& pair : m_data->speculationFailureValueProfileBuckets)
            result.add(pair.first, &pair.second);
    }

    return result;
}

void LazyOperandValueProfileParser::initialize(CompressedLazyValueProfileHolder& holder)
{
    ASSERT(m_map.isEmpty());

    if (!holder.m_data)
        return;

    for (auto& profile : holder.m_data->operandValueProfiles)
        m_map.add(profile.key(), &profile);
}

LazyOperandValueProfile* LazyOperandValueProfileParser::getIfPresent(const LazyOperandValueProfileKey& key) const
{
    auto iter = m_map.find(key);

    if (iter == m_map.end())
        return nullptr;

    return iter->value;
}

SpeculatedType LazyOperandValueProfileParser::prediction(const ConcurrentJSLocker& locker, const LazyOperandValueProfileKey& key) const
{
    LazyOperandValueProfile* profile = getIfPresent(key);
    if (!profile)
        return SpecNone;

    return profile->computeUpdatedPrediction(locker);
}

} // namespace JSC

