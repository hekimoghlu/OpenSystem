/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

#include "ComputedEffectTiming.h"
#include "InspectorWebAgentBase.h"
#include "Timer.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashMap.h>

namespace WebCore {

class AnimationEffect;
class Element;
class Event;
class KeyframeEffect;
class LocalFrame;
class Page;
class StyleOriginatedAnimation;
class WebAnimation;
class WeakPtrImplWithEventTargetData;

struct Styleable;

class InspectorAnimationAgent final : public InspectorAgentBase, public Inspector::AnimationBackendDispatcherHandler, public CanMakeCheckedPtr<InspectorAnimationAgent> {
    WTF_MAKE_NONCOPYABLE(InspectorAnimationAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorAnimationAgent);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InspectorAnimationAgent);
public:
    InspectorAnimationAgent(PageAgentContext&);
    ~InspectorAnimationAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // AnimationBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<RefPtr<Inspector::Protocol::Animation::Effect>> requestEffect(const Inspector::Protocol::Animation::AnimationId&);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::DOM::Styleable>> requestEffectTarget(const Inspector::Protocol::Animation::AnimationId&);
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::Runtime::RemoteObject>> resolveAnimation(const Inspector::Protocol::Animation::AnimationId&, const String& objectGroup);
    Inspector::Protocol::ErrorStringOr<void> startTracking();
    Inspector::Protocol::ErrorStringOr<void> stopTracking();

    // InspectorInstrumentation
    void willApplyKeyframeEffect(const Styleable&, KeyframeEffect&, const ComputedEffectTiming&);
    void didChangeWebAnimationName(WebAnimation&);
    void didSetWebAnimationEffect(WebAnimation&);
    void didChangeWebAnimationEffectTiming(WebAnimation&);
    void didChangeWebAnimationEffectTarget(WebAnimation&);
    void didCreateWebAnimation(WebAnimation&);
    void willDestroyWebAnimation(WebAnimation&);
    void frameNavigated(LocalFrame&);

private:
    String findAnimationId(WebAnimation&);
    WebAnimation* assertAnimation(Inspector::Protocol::ErrorString&, const String& animationId);
    void bindAnimation(WebAnimation&, RefPtr<Inspector::Protocol::Console::StackTrace> backtrace);
    void animationBindingTimerFired();
    void unbindAnimation(const String& animationId);
    void animationDestroyedTimerFired();
    void reset();

    void stopTrackingStyleOriginatedAnimation(StyleOriginatedAnimation&);

    std::unique_ptr<Inspector::AnimationFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::AnimationBackendDispatcher> m_backendDispatcher;

    Inspector::InjectedScriptManager& m_injectedScriptManager;
    WeakRef<Page> m_inspectedPage;

    MemoryCompactRobinHoodHashMap<Inspector::Protocol::Animation::AnimationId, WebAnimation*> m_animationIdMap;

    WeakHashMap<WebAnimation, Ref<Inspector::Protocol::Console::StackTrace>, WeakPtrImplWithEventTargetData> m_animationsPendingBinding;
    Timer m_animationBindingTimer;

    Vector<Inspector::Protocol::Animation::AnimationId> m_removedAnimationIds;
    Timer m_animationDestroyedTimer;

    struct TrackedStyleOriginatedAnimationData {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        Inspector::Protocol::Animation::AnimationId trackingAnimationId;
        ComputedEffectTiming lastComputedTiming;
    };
    HashMap<StyleOriginatedAnimation*, UniqueRef<TrackedStyleOriginatedAnimationData>> m_trackedStyleOriginatedAnimationData;

    WeakHashSet<WebAnimation, WeakPtrImplWithEventTargetData> m_animationsIgnoringEffectChanges;
    WeakHashSet<WebAnimation, WeakPtrImplWithEventTargetData> m_animationsIgnoringTargetChanges;
};

} // namespace WebCore
