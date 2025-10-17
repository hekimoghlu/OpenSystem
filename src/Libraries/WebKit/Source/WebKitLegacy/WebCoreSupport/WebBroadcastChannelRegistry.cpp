/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#include "WebBroadcastChannelRegistry.h"

#include <WebCore/BroadcastChannel.h>
#include <WebCore/SerializedScriptValue.h>
#include <wtf/CallbackAggregator.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

Ref<WebBroadcastChannelRegistry> WebBroadcastChannelRegistry::getOrCreate(bool privateSession)
{
    static NeverDestroyed<WeakPtr<WebBroadcastChannelRegistry>> defaultSessionRegistry;
    static NeverDestroyed<WeakPtr<WebBroadcastChannelRegistry>> privateSessionRegistry;
    auto& existingRegistry = privateSession ? privateSessionRegistry : defaultSessionRegistry;
    if (existingRegistry.get())
        return *existingRegistry.get();

    auto registry = adoptRef(*new WebBroadcastChannelRegistry);
    existingRegistry.get() = registry;
    return registry;
}

void WebBroadcastChannelRegistry::registerChannel(const WebCore::PartitionedSecurityOrigin& origin, const String& name, WebCore::BroadcastChannelIdentifier identifier)
{
    ASSERT(isMainThread());
    auto& channelsForOrigin = m_channels.ensure(origin, [] { return NameToChannelIdentifiersMap { }; }).iterator->value;
    auto& channelsForName = channelsForOrigin.ensure(name, [] { return Vector<WebCore::BroadcastChannelIdentifier> { }; }).iterator->value;
    ASSERT(!channelsForName.contains(identifier));
    channelsForName.append(identifier);
}

void WebBroadcastChannelRegistry::unregisterChannel(const WebCore::PartitionedSecurityOrigin& origin, const String& name, WebCore::BroadcastChannelIdentifier identifier)
{
    ASSERT(isMainThread());
    auto channelsForOriginIterator = m_channels.find(origin);
    ASSERT(channelsForOriginIterator != m_channels.end());
    if (channelsForOriginIterator == m_channels.end())
        return;
    auto channelsForNameIterator = channelsForOriginIterator->value.find(name);
    ASSERT(channelsForNameIterator != channelsForOriginIterator->value.end());
    ASSERT(channelsForNameIterator->value.contains(identifier));
    channelsForNameIterator->value.removeFirst(identifier);
}

void WebBroadcastChannelRegistry::postMessage(const WebCore::PartitionedSecurityOrigin& origin, const String& name, WebCore::BroadcastChannelIdentifier source, Ref<WebCore::SerializedScriptValue>&& message, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(isMainThread());
    auto callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));

    auto channelsForOriginIterator = m_channels.find(origin);
    ASSERT(channelsForOriginIterator != m_channels.end());
    if (channelsForOriginIterator == m_channels.end())
        return;

    auto channelsForNameIterator = channelsForOriginIterator->value.find(name);
    ASSERT(channelsForNameIterator != channelsForOriginIterator->value.end());
    for (auto& channelIdentifier : channelsForNameIterator->value) {
        if (channelIdentifier == source)
            continue;
        WebCore::BroadcastChannel::dispatchMessageTo(channelIdentifier, message.copyRef(), [callbackAggregator] { });
    }
}
