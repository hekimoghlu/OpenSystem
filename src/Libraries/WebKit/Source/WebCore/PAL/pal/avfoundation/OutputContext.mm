/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#include "OutputContext.h"

#if USE(AVFOUNDATION)

#include "OutputDevice.h"
#include <mutex>
#include <pal/spi/cocoa/AVFoundationSPI.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringConcatenate.h>

#include <pal/cocoa/AVFoundationSoftLink.h>

namespace PAL {

OutputContext::OutputContext(RetainPtr<AVOutputContext>&& context)
    : m_context(WTFMove(context))
{
}

std::optional<OutputContext>& OutputContext::sharedAudioPresentationOutputContext()
{
    static NeverDestroyed<std::optional<OutputContext>> sharedAudioPresentationOutputContext = [] () -> std::optional<OutputContext> {
        if (![PAL::getAVOutputContextClass() respondsToSelector:@selector(sharedAudioPresentationOutputContext)])
            return std::nullopt;

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
        AVOutputContext* context = [getAVOutputContextClass() sharedSystemAudioContext];
#else
        auto context = [getAVOutputContextClass() sharedAudioPresentationOutputContext];
#endif
        if (!context)
            return std::nullopt;

        return OutputContext(retainPtr(context));
    }();
    return sharedAudioPresentationOutputContext;
}

bool OutputContext::supportsMultipleOutputDevices()
{
    return [m_context respondsToSelector:@selector(supportsMultipleOutputDevices)]
        && [m_context respondsToSelector:@selector(outputDevices)]
        && [m_context supportsMultipleOutputDevices];
}

String OutputContext::deviceName()
{
    if (!supportsMultipleOutputDevices())
        return [m_context deviceName];

    return makeString(interleave(outputDevices(), [](auto& device) {
        return device.name();
    }, " + "_s));
}

Vector<OutputDevice> OutputContext::outputDevices() const
{
    if (![m_context respondsToSelector:@selector(outputDevices)]) {
        if (auto *outputDevice = [m_context outputDevice])
            return { retainPtr(outputDevice) };
        return { };
    }

    auto *avOutputDevices = [m_context outputDevices];
    return Vector<OutputDevice>(avOutputDevices.count, [&](size_t i) {
        return OutputDevice { retainPtr((AVOutputDevice *)avOutputDevices[i]) };
    });
}


}

#endif
