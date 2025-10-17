/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include <WebCore/BroadcastChannelRegistry.h>
#include <WebCore/PartitionedSecurityOrigin.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

#pragma once

class WebBroadcastChannelRegistry : public WebCore::BroadcastChannelRegistry, public CanMakeWeakPtr<WebBroadcastChannelRegistry> {
public:
    static Ref<WebBroadcastChannelRegistry> getOrCreate(bool privateSession);

    void registerChannel(const WebCore::PartitionedSecurityOrigin&, const String& name, WebCore::BroadcastChannelIdentifier) final;
    void unregisterChannel(const WebCore::PartitionedSecurityOrigin&, const String& name, WebCore::BroadcastChannelIdentifier) final;
    void postMessage(const WebCore::PartitionedSecurityOrigin&, const String& name, WebCore::BroadcastChannelIdentifier source, Ref<WebCore::SerializedScriptValue>&&, CompletionHandler<void()>&&) final;

private:
    WebBroadcastChannelRegistry() = default;

    using NameToChannelIdentifiersMap = HashMap<String, Vector<WebCore::BroadcastChannelIdentifier>>;
    HashMap<WebCore::PartitionedSecurityOrigin, NameToChannelIdentifiersMap> m_channels;
};
