/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#include "config.h"
#include "OpenXRInput.h"

#if ENABLE(WEBXR) && USE(OPENXR)
#include "OpenXRInputSource.h"
#include <wtf/TZoneMallocInlines.h>

using namespace WebCore;

namespace PlatformXR {

WTF_MAKE_TZONE_ALLOCATED_IMPL(OpenXRInput);

std::unique_ptr<OpenXRInput> OpenXRInput::create(XrInstance instance, XrSession session, XrSpace space)
{
    auto input = std::unique_ptr<OpenXRInput>(new OpenXRInput(instance, session, space));
    if (XR_FAILED(input->initialize()))
        return nullptr;
    return input;
}

OpenXRInput::OpenXRInput(XrInstance instance, XrSession session, XrSpace space)
    : m_instance(instance)
    , m_session(session)
    , m_localSpace(space)
{
}

XrResult OpenXRInput::initialize()
{
    std::array<XRHandedness, 2> hands {
        XRHandedness::Left, XRHandedness::Right
    };

    for (auto handedness : hands) {
        m_handleIndex++;;
        if (auto inputSource = OpenXRInputSource::create(m_instance, m_session, handedness, m_handleIndex))
            m_inputSources.append(makeUniqueRefFromNonNullUniquePtr(WTFMove(inputSource)));
    }

    OpenXRInputSource::SuggestedBindings bindings;
    Vector<XrActionSet> actionSets;
    for (auto& input : m_inputSources) {
        input->suggestBindings(bindings);
        actionSets.append(input->actionSet());
    }
    
    for (auto& binding : bindings) {
        auto suggestedBinding = createStructure<XrInteractionProfileSuggestedBinding, XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING>();
        RETURN_RESULT_IF_FAILED(xrStringToPath(m_instance, binding.key, &suggestedBinding.interactionProfile), m_instance);
        suggestedBinding.countSuggestedBindings = binding.value.size();
        suggestedBinding.suggestedBindings = binding.value.data();
        RETURN_RESULT_IF_FAILED(xrSuggestInteractionProfileBindings(m_instance, &suggestedBinding), m_instance);
    }

    auto attachInfo = createStructure<XrSessionActionSetsAttachInfo, XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO>();
    attachInfo.countActionSets = actionSets.size();
    attachInfo.actionSets = actionSets.data();
    RETURN_RESULT_IF_FAILED(xrAttachSessionActionSets(m_session, &attachInfo), m_instance);

    return XR_SUCCESS;
}

Vector<FrameData::InputSource> OpenXRInput::collectInputSources(const XrFrameState& frameState) const
{
    Vector<XrActiveActionSet> actionSets;
    for (auto& input : m_inputSources)
        actionSets.append(XrActiveActionSet { input->actionSet(), XR_NULL_PATH });

    auto syncInfo = createStructure<XrActionsSyncInfo, XR_TYPE_ACTIONS_SYNC_INFO>();
    syncInfo.countActiveActionSets = actionSets.size();
    syncInfo.activeActionSets = actionSets.data();
    RETURN_IF_FAILED(xrSyncActions(m_session, &syncInfo), "xrSyncActions", m_instance, { });

    Vector<FrameData::InputSource> result;
    for (auto& input : m_inputSources) {
        if (auto data = input->getInputSource(m_localSpace, frameState))
            result.append(*data);
    }

    return result;
}

void OpenXRInput::updateInteractionProfile()
{
    for (auto& input : m_inputSources)
        input->updateInteractionProfile();
}

} // namespace PlatformXR

#endif // ENABLE(WEBXR) && USE(OPENXR)
