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

#include "QualifiedName.h"
#include "SMILTime.h"
#include "Timer.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
    
class SVGElement;
class SVGSMILElement;
class SVGSVGElement;
class WeakPtrImplWithEventTargetData;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(SMILTimeContainer);
class SMILTimeContainer final : public RefCounted<SMILTimeContainer>  {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(SMILTimeContainer);
public:
    static Ref<SMILTimeContainer> create(SVGSVGElement& owner) { return adoptRef(*new SMILTimeContainer(owner)); }

    void schedule(SVGSMILElement*, SVGElement*, const QualifiedName&);
    void unschedule(SVGSMILElement*, SVGElement*, const QualifiedName&);
    void notifyIntervalsChanged();

    WEBCORE_EXPORT Seconds animationFrameDelay() const;

    SMILTime elapsed() const;

    bool isActive() const;
    bool isPaused() const;

    void begin();
    void pause();
    void resume();
    void setElapsed(SMILTime);

    void setDocumentOrderIndexesDirty() { m_documentOrderIndexesDirty = true; }

private:
    SMILTimeContainer(SVGSVGElement& owner);

    bool isStarted() const;
    void timerFired();
    void startTimer(SMILTime elapsed, SMILTime fireTime, SMILTime minimumDelay = 0);
    void updateAnimations(SMILTime elapsed, bool seekToTime = false);

    using ElementAttributePair = std::pair<SVGElement*, QualifiedName>;
    using AnimationsVector = Vector<SVGSMILElement*>;
    using GroupedAnimationsMap = UncheckedKeyHashMap<ElementAttributePair, AnimationsVector>;

    void processScheduledAnimations(const Function<void(SVGSMILElement&)>&);
    void updateDocumentOrderIndexes();
    void sortByPriority(AnimationsVector& smilElements, SMILTime elapsed);

    MonotonicTime m_beginTime;
    MonotonicTime m_pauseTime;
    Seconds m_accumulatedActiveTime { 0_s };
    MonotonicTime m_resumeTime;
    Seconds m_presetStartTime { 0_s };

    bool m_documentOrderIndexesDirty { false };
    Timer m_timer;
    GroupedAnimationsMap m_scheduledAnimations;
    WeakRef<SVGSVGElement, WeakPtrImplWithEventTargetData> m_ownerSVGElement;
};

} // namespace WebCore
