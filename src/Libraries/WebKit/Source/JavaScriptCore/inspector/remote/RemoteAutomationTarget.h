/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteControllableTarget.h"
#include <wtf/text/WTFString.h>

namespace Inspector {

class FrontendChannel;

class RemoteAutomationTarget : public RemoteControllableTarget {
public:
    JS_EXPORT_PRIVATE RemoteAutomationTarget();
    JS_EXPORT_PRIVATE ~RemoteAutomationTarget() override;

    bool isPaired() const { return m_paired; }
    JS_EXPORT_PRIVATE void setIsPaired(bool);

    bool isPendingTermination() const { return m_pendingTermination; }
    void setIsPendingTermination() { m_pendingTermination = true; }

    virtual String name() const = 0;
    RemoteControllableTarget::Type type() const override { return RemoteControllableTarget::Type::Automation; }
    bool remoteControlAllowed() const override { return !m_paired; };

private:
    bool m_paired { false };
    bool m_pendingTermination { false };
};

} // namespace Inspector

SPECIALIZE_TYPE_TRAITS_CONTROLLABLE_TARGET(Inspector::RemoteAutomationTarget, Automation)

#endif // ENABLE(REMOTE_INSPECTOR)
