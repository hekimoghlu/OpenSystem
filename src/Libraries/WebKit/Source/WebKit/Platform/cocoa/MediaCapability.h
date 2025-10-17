/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

#if ENABLE(EXTENSION_CAPABILITIES)

#include "ExtensionCapability.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS BEMediaEnvironment;

namespace WebKit {
class MediaCapability;
}

namespace WebKit {

class ExtensionCapabilityGrant;

class MediaCapability final : public ExtensionCapability, public RefCountedAndCanMakeWeakPtr<MediaCapability> {
    WTF_MAKE_NONCOPYABLE(MediaCapability);
public:
    static Ref<MediaCapability> create(URL&&);

    enum class State : uint8_t {
        Inactive,
        Activating,
        Active,
        Deactivating,
    };

    State state() const { return m_state; }
    void setState(State state) { m_state = state; }
    bool isActivatingOrActive() const;

    const URL& webPageURL() const { return m_webPageURL; }

    // ExtensionCapability
    String environmentIdentifier() const final;

    BEMediaEnvironment *platformMediaEnvironment() const { return m_mediaEnvironment.get(); }

private:
    explicit MediaCapability(URL&&);

    State m_state { State::Inactive };
    URL m_webPageURL;
    RetainPtr<BEMediaEnvironment> m_mediaEnvironment;
};

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
