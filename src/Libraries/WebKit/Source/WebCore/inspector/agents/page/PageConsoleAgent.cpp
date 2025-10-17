/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
#include "PageConsoleAgent.h"

#include "CommandLineAPIHost.h"
#include "InspectorDOMAgent.h"
#include "InstrumentingAgents.h"
#include "LogInitialization.h"
#include "Logging.h"
#include "Node.h"
#include "Page.h"
#include "WebInjectedScriptManager.h"
#include <JavaScriptCore/ConsoleMessage.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageConsoleAgent);

PageConsoleAgent::PageConsoleAgent(PageAgentContext& context)
    : WebConsoleAgent(context)
    , m_inspectedPage(context.inspectedPage)
{
}

PageConsoleAgent::~PageConsoleAgent() = default;

Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::Console::Channel>>> PageConsoleAgent::getLoggingChannels()
{
    auto channels = JSON::ArrayOf<Inspector::Protocol::Console::Channel>::create();

    auto addLogChannel = [&] (Inspector::Protocol::Console::ChannelSource source) {
        auto* logChannel = getLogChannel(Inspector::Protocol::Helpers::getEnumConstantValue(source));
        if (!logChannel)
            return;

        auto level = Inspector::Protocol::Console::ChannelLevel::Off;
        if (logChannel->state != WTFLogChannelState::Off) {
            switch (logChannel->level) {
            case WTFLogLevel::Always:
            case WTFLogLevel::Error:
            case WTFLogLevel::Warning:
            case WTFLogLevel::Info:
                level = Inspector::Protocol::Console::ChannelLevel::Basic;
                break;

            case WTFLogLevel::Debug:
                level = Inspector::Protocol::Console::ChannelLevel::Verbose;
                break;
            }
        }

        auto channel = Inspector::Protocol::Console::Channel::create()
            .setSource(source)
            .setLevel(level)
            .release();
        channels->addItem(WTFMove(channel));
    };
    addLogChannel(Inspector::Protocol::Console::ChannelSource::XML);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::JavaScript);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Network);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::ConsoleAPI);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Storage);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Appcache);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Rendering);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::CSS);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Security);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::ContentBlocker);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Media);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::MediaSource);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::WebRTC);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::ITPDebug);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::PrivateClickMeasurement);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::PaymentRequest);
    addLogChannel(Inspector::Protocol::Console::ChannelSource::Other);

    return channels;
}

Inspector::Protocol::ErrorStringOr<void> PageConsoleAgent::setLoggingChannelLevel(Inspector::Protocol::Console::ChannelSource source, Inspector::Protocol::Console::ChannelLevel level)
{
    switch (level) {
    case Inspector::Protocol::Console::ChannelLevel::Off:
        m_inspectedPage->configureLoggingChannel(Inspector::Protocol::Helpers::getEnumConstantValue(source), WTFLogChannelState::Off, WTFLogLevel::Error);
        return { };

    case Inspector::Protocol::Console::ChannelLevel::Basic:
        m_inspectedPage->configureLoggingChannel(Inspector::Protocol::Helpers::getEnumConstantValue(source), WTFLogChannelState::On, WTFLogLevel::Info);
        return { };

    case Inspector::Protocol::Console::ChannelLevel::Verbose:
        m_inspectedPage->configureLoggingChannel(Inspector::Protocol::Helpers::getEnumConstantValue(source), WTFLogChannelState::On, WTFLogLevel::Debug);
        return { };
    }

    ASSERT_NOT_REACHED();
    return { };
}

} // namespace WebCore
