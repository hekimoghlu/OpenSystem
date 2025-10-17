/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#import "WKFoundation.h"

#if ENABLE(REMOTE_INSPECTOR)

#import "APIAutomationClient.h"
#import <JavaScriptCore/RemoteInspector.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class WKProcessPool;

@protocol _WKAutomationDelegate;

namespace WebKit {

class AutomationClient final : public API::AutomationClient, Inspector::RemoteInspector::Client {
    WTF_MAKE_TZONE_ALLOCATED(AutomationClient);
public:
    explicit AutomationClient(WKProcessPool *, id <_WKAutomationDelegate>);
    virtual ~AutomationClient();

private:
    // API::AutomationClient
    bool allowsRemoteAutomation(WebProcessPool*) final { return remoteAutomationAllowed(); }
    void didRequestAutomationSession(WebKit::WebProcessPool*, const String& sessionIdentifier) final;

    // RemoteInspector::Client
    bool remoteAutomationAllowed() const final;
    void requestAutomationSession(const String& sessionIdentifier, const Inspector::RemoteInspector::Client::SessionCapabilities&) final;
    void requestedDebuggablesToWakeUp() final;
    String browserName() const final;
    String browserVersion() const final;

    WKProcessPool *m_processPool;
    WeakObjCPtr<id <_WKAutomationDelegate>> m_delegate;

    struct {
        bool allowsRemoteAutomation : 1;
        bool requestAutomationSession : 1;
        bool requestedDebuggablesToWakeUp : 1;
        bool browserNameForAutomation : 1;
        bool browserVersionForAutomation : 1;
    } m_delegateMethods;
};

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
