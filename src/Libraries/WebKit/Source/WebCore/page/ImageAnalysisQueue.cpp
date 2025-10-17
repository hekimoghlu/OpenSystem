/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#include "ImageAnalysisQueue.h"

#if ENABLE(IMAGE_ANALYSIS)

#include "Chrome.h"
#include "ChromeClient.h"
#include "HTMLCollection.h"
#include "HTMLImageElement.h"
#include "ImageOverlay.h"
#include "LocalFrameView.h"
#include "RenderImage.h"
#include "RenderView.h"
#include "TextRecognitionOptions.h"
#include "Timer.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <pal/HysteresisActivity.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static constexpr unsigned maximumPendingImageAnalysisCount = 5;
static constexpr float minimumWidthForAnalysis = 20;
static constexpr float minimumHeightForAnalysis = 20;
static constexpr Seconds resumeProcessingDelay = 100_ms;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageAnalysisQueue);

Ref<ImageAnalysisQueue> ImageAnalysisQueue::create(Page& page)
{
    return adoptRef(*new ImageAnalysisQueue(page));
}

ImageAnalysisQueue::ImageAnalysisQueue(Page& page)
    : m_page(page)
    , m_resumeProcessingTimer(*this, &ImageAnalysisQueue::resumeProcessing)
{
}

ImageAnalysisQueue::~ImageAnalysisQueue() = default;

void ImageAnalysisQueue::enqueueIfNeeded(HTMLImageElement& element)
{
    CheckedPtr renderer = downcast<RenderImage>(element.renderer());
    if (!renderer)
        return;

    CachedResourceHandle cachedImage = renderer->cachedImage();
    if (!cachedImage || cachedImage->errorOccurred())
        return;

    RefPtr image = cachedImage->image();
    if (!image || image->width() < minimumWidthForAnalysis || image->height() < minimumHeightForAnalysis)
        return;

    bool shouldAddToQueue = [&] {
        auto url = cachedImage->url();
        auto iterator = m_queuedElements.find(element);
        if (iterator == m_queuedElements.end()) {
            m_queuedElements.add(element, url);
            return true;
        }

        if (iterator->value == url)
            return false;

        iterator->value = url;

        for (auto& entry : m_queue) {
            if (entry.element == &element)
                return false;
        }

        return true;
    }();

    if (!shouldAddToQueue)
        return;

    Ref view = renderer->view().frameView();
    m_queue.enqueue({
        element,
        renderer->isVisibleInDocumentRect(view->windowToContents(view->windowClipRect())) ? Priority::High : Priority::Low,
        nextTaskNumber()
    });
    resumeProcessingSoon();
}

void ImageAnalysisQueue::resumeProcessingSoon()
{
    if (m_queue.isEmpty() || m_resumeProcessingTimer.isActive())
        return;

    m_resumeProcessingTimer.startOneShot(resumeProcessingDelay);
}

void ImageAnalysisQueue::enqueueAllImagesIfNeeded(Document& document, const String& sourceLanguageIdentifier, const String& targetLanguageIdentifier)
{
    if (!m_page)
        return;

    if (m_analysisOfAllImagesOnPageHasStarted)
        return;

    m_analysisOfAllImagesOnPageHasStarted = true;

    if (sourceLanguageIdentifier != m_sourceLanguageIdentifier || targetLanguageIdentifier != m_targetLanguageIdentifier)
        clear();

    m_sourceLanguageIdentifier = sourceLanguageIdentifier;
    m_targetLanguageIdentifier = targetLanguageIdentifier;
    enqueueAllImagesRecursive(document);
}

void ImageAnalysisQueue::enqueueAllImagesRecursive(Document& document)
{
    for (auto& image : descendantsOfType<HTMLImageElement>(document))
        enqueueIfNeeded(image);

    for (auto& frameOwner : descendantsOfType<HTMLFrameOwnerElement>(document)) {
        if (RefPtr contentDocument = frameOwner.contentDocument())
            enqueueAllImagesRecursive(*contentDocument);
    }
}

void ImageAnalysisQueue::resumeProcessing()
{
    if (!m_page)
        return;

    while (!m_queue.isEmpty() && m_pendingRequestCount < maximumPendingImageAnalysisCount) {
        RefPtr element = m_queue.dequeue().element.get();
        if (!element || !element->isConnected())
            continue;

        m_pendingRequestCount++;
        m_page->resetTextRecognitionResult(*element);

        if (auto* image = element->cachedImage(); image && !image->errorOccurred())
            m_queuedElements.set(*element, image->url());

        auto allowSnapshots = m_targetLanguageIdentifier.isEmpty() ? TextRecognitionOptions::AllowSnapshots::Yes : TextRecognitionOptions::AllowSnapshots::No;
        m_page->chrome().client().requestTextRecognition(*element, { m_sourceLanguageIdentifier, m_targetLanguageIdentifier, allowSnapshots }, [this, page = m_page] (auto&&) {
            if (!page || page->imageAnalysisQueueIfExists() != this)
                return;

            if (m_pendingRequestCount)
                m_pendingRequestCount--;

            resumeProcessingSoon();

            if (m_queue.isEmpty() && m_imageQueueEmptyHysteresis)
                m_imageQueueEmptyHysteresis->impulse();
        });
    }
}

void ImageAnalysisQueue::setDidBecomeEmptyCallback(Function<void()>&& callback)
{
    m_imageQueueEmptyHysteresis = makeUnique<PAL::HysteresisActivity>([callback = WTFMove(callback)] (PAL::HysteresisState state) {
        if (state == PAL::HysteresisState::Stopped)
            callback();
    }, 1_s);
}

void ImageAnalysisQueue::clearDidBecomeEmptyCallback()
{
    m_imageQueueEmptyHysteresis = nullptr;
}

void ImageAnalysisQueue::clear()
{
    // FIXME: This should cancel pending requests in addition to emptying the task queue.
    m_pendingRequestCount = 0;
    m_resumeProcessingTimer.stop();
    m_queue = { };
    m_queuedElements.clear();
    m_sourceLanguageIdentifier = { };
    m_targetLanguageIdentifier = { };
    m_currentTaskNumber = 0;
    m_analysisOfAllImagesOnPageHasStarted = false;
    m_imageQueueEmptyHysteresis = nullptr;
}

} // namespace WebCore

#endif // ENABLE(IMAGE_ANALYSIS)
