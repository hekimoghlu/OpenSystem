/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include "CommandLineAPIHost.h"
#include <JavaScriptCore/InjectedScriptManager.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class LocalDOMWindow;

class WebInjectedScriptManager final : public Inspector::InjectedScriptManager {
    WTF_MAKE_NONCOPYABLE(WebInjectedScriptManager);
    WTF_MAKE_TZONE_ALLOCATED(WebInjectedScriptManager);
public:
    WebInjectedScriptManager(Inspector::InspectorEnvironment&, Ref<Inspector::InjectedScriptHost>&&);
    ~WebInjectedScriptManager() override = default;

    const RefPtr<CommandLineAPIHost>& commandLineAPIHost() const { return m_commandLineAPIHost; }

    void connect() override;
    void disconnect() override;
    void discardInjectedScripts() override;

    void discardInjectedScriptsFor(LocalDOMWindow&);

private:
    void didCreateInjectedScript(const Inspector::InjectedScript&) override;

    RefPtr<CommandLineAPIHost> m_commandLineAPIHost;
};

} // namespace WebCore
