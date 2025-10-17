/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

#include "WebSocketExtensionProcessor.h"
#include <wtf/Vector.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WebSocketExtensionDispatcher {
public:
    WebSocketExtensionDispatcher() { }

    void reset();
    void addProcessor(std::unique_ptr<WebSocketExtensionProcessor>);
    const String createHeaderValue() const;

    bool processHeaderValue(const String&);
    String acceptedExtensions() const;
    String failureReason() const;

private:
    void appendAcceptedExtension(const String& extensionToken, HashMap<String, String>& extensionParameters);
    void fail(const String& reason);

    Vector<std::unique_ptr<WebSocketExtensionProcessor>> m_processors;
    StringBuilder m_acceptedExtensionsBuilder;
    String m_failureReason;
};

} // namespace WebCore
