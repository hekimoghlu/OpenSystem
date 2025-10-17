/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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

// FIXME: This should be in PlatformHave.h instead of here.
// FIXME: Should be able to do this even without the Apple internal SDK.
// FIXME: Is it correct to that we do not have this on tvOS?
#if USE(APPLE_INTERNAL_SDK) && PLATFORM(COCOA) && !PLATFORM(APPLETV)
#define HAVE_NW_ACTIVITY 1
#endif

#if HAVE(NW_ACTIVITY)
#include <nw/private.h>
#include <wtf/RetainPtr.h>
#endif

namespace WebKit {

class NetworkActivityTracker {
public:
    enum class Domain {
        // These are defined to match analogous values used in the Darwin implementation.
        // If they are renumbered, platform-specific code will need to be added to map
        // them to the Darwin-specific values.

        Invalid = 0,
        WebKit = 16,
    };

    enum class Label {
        // These are ours to define, but once defined, they shouldn't change. They can
        // be obsolesced and replaced with other codes, but previously-defined codes
        // should not be renumbered. Previously assigned values should not be re-used.

        Invalid = 0,
        LoadPage = 1,
        LoadResource = 2,
    };

    enum class CompletionCode {
        Undefined,
        None,
        Success,
        Failure,
        Cancel,
    };

    NetworkActivityTracker() = default;
    explicit NetworkActivityTracker(Label, Domain = Domain::WebKit);
    ~NetworkActivityTracker();

    void setParent(NetworkActivityTracker&);
    void start();
    void complete(CompletionCode);

#if HAVE(NW_ACTIVITY)
    nw_activity_t getPlatformObject() const { return m_networkActivity.get(); }
#endif

private:
#if HAVE(NW_ACTIVITY)
    Domain m_domain { Domain::Invalid };
    Label m_label { Label::Invalid };
    bool m_isCompleted { false };
    RetainPtr<nw_activity_t> m_networkActivity;
#endif
};

} // namespace WebKit
