/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#if ENABLE(WEB_AUDIO)

#include "DelayProcessor.h"
 
#include "DelayDSPKernel.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DelayProcessor);

DelayProcessor::DelayProcessor(BaseAudioContext& context, float sampleRate, unsigned numberOfChannels, double maxDelayTime)
    : AudioDSPKernelProcessor(sampleRate, numberOfChannels)
    , m_delayTime(AudioParam::create(context, "delayTime"_s, 0.0, 0.0, maxDelayTime, AutomationRate::ARate))
{

}

DelayProcessor::~DelayProcessor()
{
    if (isInitialized())
        uninitialize();
}

std::unique_ptr<AudioDSPKernel> DelayProcessor::createKernel()
{
    return makeUnique<DelayDSPKernel>(this);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
