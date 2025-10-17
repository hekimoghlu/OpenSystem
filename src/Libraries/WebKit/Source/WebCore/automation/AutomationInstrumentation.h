/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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

#if ENABLE(WEBDRIVER_BIDI)

#include <JavaScriptCore/ConsoleMessage.h>
#include <JavaScriptCore/ConsoleTypes.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Function.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefPtr.h>
#include <wtf/WallTime.h>

namespace Inspector {
class ConsoleMessage;
}

namespace WebCore {

class WEBCORE_EXPORT AutomationInstrumentationClient : public AbstractRefCountedAndCanMakeWeakPtr<AutomationInstrumentationClient> {
public:
    virtual ~AutomationInstrumentationClient() = default;

    virtual void addMessageToConsole(const JSC::MessageSource&, const JSC::MessageLevel&, const String&, const JSC::MessageType&, const WallTime&) = 0;
};


class WEBCORE_EXPORT AutomationInstrumentation {
public:
    static void setClient(const AutomationInstrumentationClient&);
    static void clearClient();

    static void addMessageToConsole(const std::unique_ptr<Inspector::ConsoleMessage>&);
};

} // namespace WebCore

#endif
