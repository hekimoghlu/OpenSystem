/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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

#include <wtf/WeakRef.h>

namespace WebKit {

class ProcessAssertion;
class ProcessThrottlerActivity;
class RemotePageProxy;
class WebPageProxy;
class WebProcessProxy;

class WebProcessActivityState {
    WTF_MAKE_FAST_ALLOCATED;
public:
    explicit WebProcessActivityState(WebPageProxy&);
    explicit WebProcessActivityState(RemotePageProxy&);

    void takeVisibleActivity();
    void takeAudibleActivity();
    void takeCapturingActivity();
    void takeMutedCaptureAssertion();

    void reset();
    void dropVisibleActivity();
    void dropAudibleActivity();
    void dropCapturingActivity();
    void dropMutedCaptureAssertion();

    bool hasValidVisibleActivity() const;
    bool hasValidAudibleActivity() const;
    bool hasValidCapturingActivity() const;
    bool hasValidMutedCaptureAssertion() const;

#if PLATFORM(IOS_FAMILY)
    void takeOpeningAppLinkActivity();
    void dropOpeningAppLinkActivity();
    bool hasValidOpeningAppLinkActivity() const;
#endif

#if ENABLE(WEB_PROCESS_SUSPENSION_DELAY)
    void updateWebProcessSuspensionDelay();
    void takeAccessibilityActivityWhenInWindow();
    void takeAccessibilityActivity();
    bool hasAccessibilityActivityForTesting() const;
    void viewDidEnterWindow();
    void viewDidLeaveWindow();
#endif

private:
    WebProcessProxy& process() const;
    Ref<WebProcessProxy> protectedProcess() const;

    std::variant<WeakRef<WebPageProxy>, WeakRef<RemotePageProxy>> m_page;

    RefPtr<ProcessThrottlerActivity> m_isVisibleActivity;
#if ENABLE(WEB_PROCESS_SUSPENSION_DELAY)
    Ref<ProcessThrottlerTimedActivity> m_wasRecentlyVisibleActivity;
    RefPtr<ProcessThrottlerActivity> m_accessibilityActivity;
    bool m_takeAccessibilityActivityWhenInWindow { false };
#endif
    RefPtr<ProcessThrottlerActivity> m_isAudibleActivity;
    RefPtr<ProcessThrottlerActivity> m_isCapturingActivity;
    RefPtr<ProcessAssertion> m_isMutedCaptureAssertion;
#if PLATFORM(IOS_FAMILY)
    RefPtr<ProcessThrottlerActivity> m_openingAppLinkActivity;
#endif
};

} // namespace WebKit
