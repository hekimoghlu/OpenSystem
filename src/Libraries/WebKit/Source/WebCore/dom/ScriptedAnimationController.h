/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#include "AnimationFrameRate.h"
#include "Document.h"
#include "ReducedResolutionSeconds.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class ImminentlyScheduledWorkScope;
class Page;
class RequestAnimationFrameCallback;
class UserGestureToken;
class WeakPtrImplWithEventTargetData;

class ScriptedAnimationController : public RefCounted<ScriptedAnimationController>
{
public:
    static Ref<ScriptedAnimationController> create(Document& document)
    {
        return adoptRef(*new ScriptedAnimationController(document));
    }
    ~ScriptedAnimationController();
    void clearDocumentPointer() { m_document = nullptr; }

    WEBCORE_EXPORT Seconds interval() const;
    WEBCORE_EXPORT OptionSet<ThrottlingReason> throttlingReasons() const;

    void suspend();
    void resume();

    void addThrottlingReason(ThrottlingReason reason) { m_throttlingReasons.add(reason); }
    void removeThrottlingReason(ThrottlingReason reason) { m_throttlingReasons.remove(reason); }

    using CallbackId = int;
    CallbackId registerCallback(Ref<RequestAnimationFrameCallback>&&);
    void cancelCallback(CallbackId);
    void serviceRequestAnimationFrameCallbacks(ReducedResolutionSeconds);

private:
    ScriptedAnimationController(Document&);

    Page* page() const;
    Seconds preferredScriptedAnimationInterval() const;
    bool isThrottledRelativeToPage() const;
    bool shouldRescheduleRequestAnimationFrame(ReducedResolutionSeconds) const;
    void scheduleAnimation();
    RefPtr<Document> protectedDocument();

    struct CallbackData {
        Ref<RequestAnimationFrameCallback> callback;
        RefPtr<UserGestureToken> userGestureTokenToForward;
        RefPtr<ImminentlyScheduledWorkScope> scheduledWorkScope;
    };
    Vector<CallbackData> m_callbackDataList;

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    CallbackId m_nextCallbackId { 0 };
    int m_suspendCount { 0 };

    ReducedResolutionSeconds m_lastAnimationFrameTimestamp;
    OptionSet<ThrottlingReason> m_throttlingReasons;
};

} // namespace WebCore
