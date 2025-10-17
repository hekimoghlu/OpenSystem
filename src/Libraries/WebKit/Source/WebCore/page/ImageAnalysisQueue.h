/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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

#if ENABLE(IMAGE_ANALYSIS)

#include "Timer.h"
#include <wtf/PriorityQueue.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/URLHash.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakPtr.h>

namespace PAL {
class HysteresisActivity;
}

namespace WebCore {

class Document;
class HTMLImageElement;
class Page;
class Timer;
class WeakPtrImplWithEventTargetData;

class ImageAnalysisQueue final : public RefCounted<ImageAnalysisQueue> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ImageAnalysisQueue, WEBCORE_EXPORT);
public:
    static Ref<ImageAnalysisQueue> create(Page&);
    WEBCORE_EXPORT ~ImageAnalysisQueue();

    WEBCORE_EXPORT void enqueueAllImagesIfNeeded(Document&, const String& sourceLanguageIdentifier, const String& targetLanguageIdentifier);
    void clear();

    void enqueueIfNeeded(HTMLImageElement&);

    WEBCORE_EXPORT void setDidBecomeEmptyCallback(Function<void()>&&);
    WEBCORE_EXPORT void clearDidBecomeEmptyCallback();

private:
    explicit ImageAnalysisQueue(Page&);

    void resumeProcessingSoon();
    void resumeProcessing();

    void enqueueAllImagesRecursive(Document&);

    enum class Priority : bool { Low, High };
    struct Task {
        WeakPtr<HTMLImageElement, WeakPtrImplWithEventTargetData> element;
        Priority priority { Priority::Low };
        unsigned taskNumber { 0 };
    };

    static bool firstIsHigherPriority(const Task&, const Task&);
    unsigned nextTaskNumber() { return ++m_currentTaskNumber; }

    // FIXME: Refactor the source and target LIDs into either a std::pair<> of strings, or its own named struct.
    String m_sourceLanguageIdentifier;
    String m_targetLanguageIdentifier;
    WeakPtr<Page> m_page;
    Timer m_resumeProcessingTimer;
    WeakHashMap<HTMLImageElement, URL, WeakPtrImplWithEventTargetData> m_queuedElements;
    PriorityQueue<Task, firstIsHigherPriority> m_queue;
    unsigned m_pendingRequestCount { 0 };
    unsigned m_currentTaskNumber { 0 };
    std::unique_ptr<PAL::HysteresisActivity> m_imageQueueEmptyHysteresis;
    bool m_analysisOfAllImagesOnPageHasStarted { false };
};

inline bool ImageAnalysisQueue::firstIsHigherPriority(const Task& first, const Task& second)
{
    if (first.priority != second.priority)
        return first.priority == Priority::High;

    return first.taskNumber < second.taskNumber;
}

} // namespace WebCore

#endif // ENABLE(IMAGE_ANALYSIS)
