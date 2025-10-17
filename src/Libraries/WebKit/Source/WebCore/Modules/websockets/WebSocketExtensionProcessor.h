/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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

#include <wtf/HashMap.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WebSocketExtensionProcessor {
public:
    virtual ~WebSocketExtensionProcessor() = default;

    String extensionToken() const { return m_extensionToken; }

    // The return value of this method will be a part of the value of
    // Sec-WebSocket-Extensions.
    virtual String handshakeString() = 0;

    // This should validate the server's response parameters which are passed
    // as HashMap<key, value>. This may also do something for the extension.
    // Note that this method may be called more than once when the server
    // response contains duplicate extension token that matches extensionToken().
    virtual bool processResponse(const HashMap<String, String>&) = 0;

    // If procecssResponse() returns false, this should provide the reason.
    virtual String failureReason() { return makeString("Extension "_s, m_extensionToken, " failed"_s); }

protected:
    explicit WebSocketExtensionProcessor(const String& extensionToken)
        : m_extensionToken(extensionToken)
    {
    }

private:
    String m_extensionToken;
};

} // namespace WebCore
