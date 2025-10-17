/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#pragma once

#include "APIContentWorld.h"
#include "ScriptMessageHandlerIdentifier.h"
#include "WebUserContentControllerDataTypes.h"
#include <wtf/Identified.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SecurityOriginData;
class SerializedScriptValue;
}

namespace API {
class ContentWorld;
class SerializedScriptValue;
}

namespace WebKit {

class WebPageProxy;
class WebFrameProxy;
struct FrameInfoData;

class WebScriptMessageHandler : public RefCounted<WebScriptMessageHandler>, public Identified<ScriptMessageHandlerIdentifier>  {
public:
    class Client {
    public:
        virtual ~Client() { }
        virtual void didPostMessage(WebPageProxy&, FrameInfoData&&, API::ContentWorld&, WebCore::SerializedScriptValue&) = 0;
        virtual bool supportsAsyncReply() = 0;
        virtual void didPostMessageWithAsyncReply(WebPageProxy&, FrameInfoData&&, API::ContentWorld&, WebCore::SerializedScriptValue&, WTF::Function<void(API::SerializedScriptValue*, const String&)>&&) = 0;
    };

    static Ref<WebScriptMessageHandler> create(std::unique_ptr<Client>, const String& name, API::ContentWorld&);
    virtual ~WebScriptMessageHandler();

    String name() const { return m_name; }

    API::ContentWorld& world() { return m_world.get(); }
    Ref<API::ContentWorld> protectedWorld();

    Client& client() const { return *m_client; }

private:
    WebScriptMessageHandler(std::unique_ptr<Client>, const String&, API::ContentWorld&);

    std::unique_ptr<Client> m_client;
    String m_name;
    Ref<API::ContentWorld> m_world;
};

} // namespace API
