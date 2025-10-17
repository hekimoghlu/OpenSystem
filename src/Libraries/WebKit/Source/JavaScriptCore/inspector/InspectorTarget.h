/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include "InspectorFrontendChannel.h"
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class InspectorTarget;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<Inspector::InspectorTarget> : std::true_type { };
}

namespace Inspector {

// FIXME: Add DedicatedWorker Inspector Targets
// FIXME: Add ServiceWorker Inspector Targets
enum class InspectorTargetType : uint8_t {
    Page,
    DedicatedWorker,
    ServiceWorker,
};

class InspectorTarget : public CanMakeWeakPtr<InspectorTarget> {
public:
    virtual ~InspectorTarget() = default;

    // State.
    virtual String identifier() const = 0;
    virtual InspectorTargetType type() const = 0;

    virtual bool isProvisional() const { return false; }
    bool isPaused() const { return m_isPaused; }
    void pause();
    JS_EXPORT_PRIVATE void resume();
    JS_EXPORT_PRIVATE void setResumeCallback(WTF::Function<void()>&&);

    // Connection management.
    virtual void connect(FrontendChannel::ConnectionType) = 0;
    virtual void disconnect() = 0;
    virtual void sendMessageToTargetBackend(const String&) = 0;

private:
    WTF::Function<void()> m_resumeCallback;
    bool m_isPaused { false };
};

} // namespace Inspector
