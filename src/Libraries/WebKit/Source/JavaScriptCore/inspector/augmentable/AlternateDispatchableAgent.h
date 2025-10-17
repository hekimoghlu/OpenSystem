/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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

#if ENABLE(INSPECTOR_ALTERNATE_DISPATCHERS)

#include "AugmentableInspectorController.h"
#include "InspectorAgentBase.h"
#include "InspectorAlternateBackendDispatchers.h"
#include <wtf/Forward.h>

namespace Inspector {

template<typename TBackendDispatcher, typename TAlternateDispatcher>
class AlternateDispatchableAgent final : public InspectorAgentBase {
    WTF_MAKE_FAST_ALLOCATED;
public:
    AlternateDispatchableAgent(const String& domainName, AugmentableInspectorController& controller, std::unique_ptr<TAlternateDispatcher> alternateDispatcher)
        : InspectorAgentBase(domainName)
        , m_alternateDispatcher(WTFMove(alternateDispatcher))
        , m_backendDispatcher(TBackendDispatcher::create(controller.backendDispatcher(), nullptr))
    {
        m_backendDispatcher->setAlternateDispatcher(m_alternateDispatcher.get());
        m_alternateDispatcher->setBackendDispatcher(&controller.backendDispatcher());
    }

    virtual ~AlternateDispatchableAgent()
    {
        m_alternateDispatcher->setBackendDispatcher(nullptr);
    }

    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final
    {
    }

    void willDestroyFrontendAndBackend(DisconnectReason) final
    {
    }

private:
    std::unique_ptr<TAlternateDispatcher> m_alternateDispatcher;
    RefPtr<TBackendDispatcher> m_backendDispatcher;
};

} // namespace Inspector

#endif // ENABLE(INSPECTOR_ALTERNATE_DISPATCHERS)
