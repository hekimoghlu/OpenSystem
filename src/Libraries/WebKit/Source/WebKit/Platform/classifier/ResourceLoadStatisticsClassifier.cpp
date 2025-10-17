/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "ResourceLoadStatisticsClassifier.h"

#include "Logging.h"
#include <WebCore/ResourceLoadStatistics.h>

namespace WebKit {
using namespace WebCore;

static double vectorLength(unsigned a, unsigned b, unsigned c)
{
    return std::hypot(a, b, c);
}

static const auto featureVectorLengthThresholdHigh = 3;
static const auto featureVectorLengthThresholdVeryHigh = 30;
ResourceLoadPrevalence ResourceLoadStatisticsClassifier::calculateResourcePrevalence(const ResourceLoadStatistics& resourceStatistic, ResourceLoadPrevalence currentPrevalence)
{
    ASSERT(currentPrevalence != VeryHigh);

    auto subresourceUnderTopFrameDomainsCount = resourceStatistic.subresourceUnderTopFrameDomains.size();
    auto subresourceUniqueRedirectsToCount = resourceStatistic.subresourceUniqueRedirectsTo.size();
    auto subframeUnderTopFrameDomainsCount = resourceStatistic.subframeUnderTopFrameDomains.size();
    auto topFrameUniqueRedirectsToCount = resourceStatistic.topFrameUniqueRedirectsTo.size();

    return calculateResourcePrevalence(subresourceUnderTopFrameDomainsCount, subresourceUniqueRedirectsToCount, subframeUnderTopFrameDomainsCount, topFrameUniqueRedirectsToCount, currentPrevalence);
}

ResourceLoadPrevalence ResourceLoadStatisticsClassifier::calculateResourcePrevalence(unsigned subresourceUnderTopFrameDomainsCount, unsigned subresourceUniqueRedirectsToCount, unsigned subframeUnderTopFrameDomainsCount, unsigned topFrameUniqueRedirectsToCount, ResourceLoadPrevalence currentPrevalence)
{
    if (!subresourceUnderTopFrameDomainsCount
        && !subresourceUniqueRedirectsToCount
        && !subframeUnderTopFrameDomainsCount
        && !topFrameUniqueRedirectsToCount) {
        return Low;
    }

    if (vectorLength(subresourceUnderTopFrameDomainsCount, subresourceUniqueRedirectsToCount, subframeUnderTopFrameDomainsCount) > featureVectorLengthThresholdVeryHigh)
        return VeryHigh;

    if (currentPrevalence == High
        || subresourceUnderTopFrameDomainsCount > featureVectorLengthThresholdHigh
        || subresourceUniqueRedirectsToCount > featureVectorLengthThresholdHigh
        || subframeUnderTopFrameDomainsCount > featureVectorLengthThresholdHigh
        || topFrameUniqueRedirectsToCount > featureVectorLengthThresholdHigh
        || classify(subresourceUnderTopFrameDomainsCount, subresourceUniqueRedirectsToCount, subframeUnderTopFrameDomainsCount))
        return High;

    return Low;
}

bool ResourceLoadStatisticsClassifier::classifyWithVectorThreshold(unsigned a, unsigned b, unsigned c)
{
    LOG(ResourceLoadStatistics, "ResourceLoadStatisticsClassifier::classifyWithVectorThreshold(): Classified with threshold.");
    return vectorLength(a, b, c) > featureVectorLengthThresholdHigh;
}
    
}
