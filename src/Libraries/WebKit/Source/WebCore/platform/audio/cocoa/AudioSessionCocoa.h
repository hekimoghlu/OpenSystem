/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#if USE(AUDIO_SESSION) && PLATFORM(COCOA)

#include "AudioSession.h"
#include <wtf/TZoneMalloc.h>

namespace WTF {
class WorkQueue;
}

namespace WebCore {

class AudioSessionCocoa : public AudioSession {
    WTF_MAKE_TZONE_ALLOCATED(AudioSessionCocoa);
public:
    virtual ~AudioSessionCocoa();

    bool isEligibleForSmartRouting() const { return m_isEligibleForSmartRouting; }

    enum class ForceUpdate : bool { No, Yes };
    void setEligibleForSmartRouting(bool, ForceUpdate = ForceUpdate::No);

protected:
    AudioSessionCocoa();

    void setEligibleForSmartRoutingInternal(bool);

    // AudioSession
    bool tryToSetActiveInternal(bool) final;
    void setCategory(CategoryType, Mode, RouteSharingPolicy) override;

    bool m_isEligibleForSmartRouting { false };
    Ref<WTF::WorkQueue> m_workQueue;
};

}

#endif
