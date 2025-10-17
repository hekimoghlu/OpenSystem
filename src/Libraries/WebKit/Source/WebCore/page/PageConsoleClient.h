/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#include <JavaScriptCore/ConsoleClient.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace Inspector {
class ConsoleMessage;
}

namespace JSC {
class CallFrame;
}

using JSC::MessageType;

namespace WebCore {

class Document;
class Page;

class WEBCORE_EXPORT PageConsoleClient final : public JSC::ConsoleClient, public CanMakeCheckedPtr<PageConsoleClient> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PageConsoleClient, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageConsoleClient);
public:
    explicit PageConsoleClient(Page&);
    virtual ~PageConsoleClient();

    static bool shouldPrintExceptions();
    static void setShouldPrintExceptions(bool);

    static void mute();
    static void unmute();

    void addMessage(std::unique_ptr<Inspector::ConsoleMessage>&&);

    // The following addMessage function are deprecated.
    // Callers should try to create the ConsoleMessage themselves.
    void addMessage(MessageSource, MessageLevel, const String& message, const String& sourceURL, unsigned lineNumber, unsigned columnNumber, RefPtr<Inspector::ScriptCallStack>&& = nullptr, JSC::JSGlobalObject* = nullptr, unsigned long requestIdentifier = 0);
    void addMessage(MessageSource, MessageLevel, const String& message, Ref<Inspector::ScriptCallStack>&&);
    void addMessage(MessageSource, MessageLevel, const String& message, unsigned long requestIdentifier = 0, Document* = nullptr);

    static void logMessageToSystemConsole(const Inspector::ConsoleMessage&);

private:
    void messageWithTypeAndLevel(MessageType, MessageLevel, JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) override;
    void count(JSC::JSGlobalObject*, const String& label) override;
    void countReset(JSC::JSGlobalObject*, const String& label) override;
    void profile(JSC::JSGlobalObject*, const String& title) override;
    void profileEnd(JSC::JSGlobalObject*, const String& title) override;
    void takeHeapSnapshot(JSC::JSGlobalObject*, const String& title) override;
    void time(JSC::JSGlobalObject*, const String& label) override;
    void timeLog(JSC::JSGlobalObject*, const String& label, Ref<Inspector::ScriptArguments>&&) override;
    void timeEnd(JSC::JSGlobalObject*, const String& label) override;
    void timeStamp(JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) override;
    void record(JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) override;
    void recordEnd(JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) override;
    void screenshot(JSC::JSGlobalObject*, Ref<Inspector::ScriptArguments>&&) override;

    Ref<Page> protectedPage() const;

    WeakRef<Page> m_page;
};

} // namespace WebCore
