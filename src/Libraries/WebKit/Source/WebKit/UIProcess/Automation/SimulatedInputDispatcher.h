/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

#if ENABLE(WEBDRIVER_ACTIONS_API)

#include <WebCore/FrameIdentifier.h>
#include <WebCore/IntPoint.h>
#include <variant>
#include <wtf/CompletionHandler.h>
#include <wtf/ListHashSet.h>
#include <wtf/RunLoop.h>
#include <wtf/Seconds.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace Inspector { namespace Protocol { namespace Automation {
enum class ErrorMessage;
enum class KeyboardInteractionType;
enum class MouseButton;
enum class MouseInteraction;
enum class MouseMoveOrigin;
enum class VirtualKey;
} } }

namespace WebKit {

class AutomationCommandError;
using AutomationCompletionHandler = WTF::CompletionHandler<void(std::optional<AutomationCommandError>)>;

class WebPageProxy;

using KeyboardInteraction = Inspector::Protocol::Automation::KeyboardInteractionType;
using VirtualKey = Inspector::Protocol::Automation::VirtualKey;
using VirtualKeyMap = HashMap<VirtualKey, VirtualKey, WTF::IntHash<VirtualKey>, WTF::StrongEnumHashTraits<VirtualKey>>;
#if ENABLE(WEBDRIVER_KEYBOARD_GRAPHEME_CLUSTERS)
// A CharKey must only ever represent a single unicode codepoint or a single grapheme cluster.
using CharKey = String;
using CharKeySet = ListHashSet<String>;
#else
using CharKey = char32_t;
using CharKeySet = ListHashSet<uint32_t>;
#endif
using MouseButton = Inspector::Protocol::Automation::MouseButton;
using MouseInteraction = Inspector::Protocol::Automation::MouseInteraction;
using MouseMoveOrigin = Inspector::Protocol::Automation::MouseMoveOrigin;

enum class SimulatedInputSourceType {
    Null, // Used to induce a minimum duration.
    Keyboard,
    Mouse,
    Touch,
    Wheel,
    Pen,
};

enum class TouchInteraction {
    TouchDown,
    MoveTo,
    LiftUp,
};

struct SimulatedInputSourceState {
    CharKeySet pressedCharKeys;
    VirtualKeyMap pressedVirtualKeys;
    std::optional<MouseButton> pressedMouseButton;
    std::optional<MouseInteraction> mouseInteraction;
    std::optional<MouseMoveOrigin> origin;
    std::optional<String> nodeHandle;
    std::optional<WebCore::IntPoint> location;
    std::optional<WebCore::IntSize> scrollDelta;
    std::optional<Seconds> duration;

    static SimulatedInputSourceState emptyStateForSourceType(SimulatedInputSourceType);
};

struct SimulatedInputSource : public RefCounted<SimulatedInputSource> {
public:
    SimulatedInputSourceType type;

    // The last state associated with this input source.
    SimulatedInputSourceState state;

    static Ref<SimulatedInputSource> create(SimulatedInputSourceType type)
    {
        return adoptRef(*new SimulatedInputSource(type));
    }

private:
    SimulatedInputSource(SimulatedInputSourceType type)
        : type(type)
        , state(SimulatedInputSourceState::emptyStateForSourceType(type))
    { }
};

struct SimulatedInputKeyFrame {
public:
    using StateEntry = std::pair<SimulatedInputSource&, SimulatedInputSourceState>;

    explicit SimulatedInputKeyFrame(Vector<StateEntry>&&);
    Seconds maximumDuration() const;

    static SimulatedInputKeyFrame keyFrameFromStateOfInputSources(const HashMap<String, Ref<SimulatedInputSource>>&);
    static SimulatedInputKeyFrame keyFrameToResetInputSources(const HashMap<String, Ref<SimulatedInputSource>>&);

    Vector<StateEntry> states;
};

class SimulatedInputDispatcher : public RefCounted<SimulatedInputDispatcher> {
    WTF_MAKE_NONCOPYABLE(SimulatedInputDispatcher);
public:
    class Client {
    public:
        virtual ~Client() { }
#if ENABLE(WEBDRIVER_MOUSE_INTERACTIONS)
        virtual void simulateMouseInteraction(WebPageProxy&, MouseInteraction, MouseButton, const WebCore::IntPoint& locationInView, const String& pointerType, AutomationCompletionHandler&&) = 0;
#endif
#if ENABLE(WEBDRIVER_TOUCH_INTERACTIONS)
        virtual void simulateTouchInteraction(WebPageProxy&, TouchInteraction, const WebCore::IntPoint& locationInView, std::optional<Seconds> duration, AutomationCompletionHandler&&) = 0;
#endif
#if ENABLE(WEBDRIVER_KEYBOARD_INTERACTIONS)
        virtual void simulateKeyboardInteraction(WebPageProxy&, KeyboardInteraction, std::variant<VirtualKey, CharKey>&&, AutomationCompletionHandler&&) = 0;
#endif
#if ENABLE(WEBDRIVER_WHEEL_INTERACTIONS)
        virtual void simulateWheelInteraction(WebPageProxy&, const WebCore::IntPoint& locationInView, const WebCore::IntSize& delta, AutomationCompletionHandler&&) = 0;
#endif
        virtual void viewportInViewCenterPointOfElement(WebPageProxy&, std::optional<WebCore::FrameIdentifier>, const String& nodeHandle, Function<void (std::optional<WebCore::IntPoint>, std::optional<AutomationCommandError>)>&&) = 0;
    };

    static Ref<SimulatedInputDispatcher> create(WebPageProxy& page, SimulatedInputDispatcher::Client& client)
    {
        return adoptRef(*new SimulatedInputDispatcher(page, client));
    }

    ~SimulatedInputDispatcher();

    void run(std::optional<WebCore::FrameIdentifier>, Vector<SimulatedInputKeyFrame>&& keyFrames, const HashMap<String, Ref<SimulatedInputSource>>& inputSources, AutomationCompletionHandler&&);
    void cancel();

    bool isActive() const;

private:
    SimulatedInputDispatcher(WebPageProxy&, SimulatedInputDispatcher::Client&);

    void transitionToNextKeyFrame();
    void transitionBetweenKeyFrames(const SimulatedInputKeyFrame&, const SimulatedInputKeyFrame&, AutomationCompletionHandler&&);

    void transitionToNextInputSourceState();
    void transitionInputSourceToState(SimulatedInputSource&, SimulatedInputSourceState& newState, AutomationCompletionHandler&&);
    void finishDispatching(std::optional<AutomationCommandError>);

    void keyFrameTransitionDurationTimerFired();
    bool isKeyFrameTransitionComplete() const;

    void resolveLocation(const WebCore::IntPoint& currentLocation, std::optional<WebCore::IntPoint> location, MouseMoveOrigin, std::optional<String> nodeHandle, Function<void (std::optional<WebCore::IntPoint>, std::optional<AutomationCommandError>)>&&);

    Ref<WebPageProxy> protectedPage() const;

    WeakRef<WebPageProxy> m_page;
    SimulatedInputDispatcher::Client& m_client;

    std::optional<WebCore::FrameIdentifier> m_frameID;
    AutomationCompletionHandler m_runCompletionHandler;
    AutomationCompletionHandler m_keyFrameTransitionCompletionHandler;
    RunLoop::Timer m_keyFrameTransitionDurationTimer;

    Vector<SimulatedInputKeyFrame> m_keyframes;

    // The position within m_keyframes.
    unsigned m_keyframeIndex { 0 };

    // The position within the input source state vector at m_keyframes[m_keyframeIndex].
    // Events that reflect input source state transitions are dispatched serially based on this order.
    unsigned m_inputSourceStateIndex { 0 };
};

} // namespace WebKit

#endif // ENABLE(WEBDRIVER_ACTIONS_API)
