/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#import "config.h"
#import "Page.h"

#import "DocumentLoader.h"
#import "FrameLoader.h"
#import "FrameTree.h"
#import "LayoutTreeBuilder.h"
#import "LocalFrame.h"
#import "Logging.h"
#import "PlatformMediaSessionManager.h"
#import "RenderObject.h"
#import "SVGDocument.h"
#import "SVGElementTypeHelpers.h"
#import <pal/Logging.h>

#if PLATFORM(IOS_FAMILY)
#import "WebCoreThreadInternal.h"
#endif

namespace WebCore {

void Page::platformInitialize()
{
#if PLATFORM(IOS_FAMILY)
    addSchedulePair(SchedulePair::create(WebThreadNSRunLoop(), kCFRunLoopCommonModes));
#else
    addSchedulePair(SchedulePair::create([[NSRunLoop currentRunLoop] getCFRunLoop], kCFRunLoopCommonModes));
#endif

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
#if ENABLE(TREE_DEBUGGING)
        PAL::registerNotifyCallback("com.apple.WebKit.showRenderTree"_s, printRenderTreeForLiveDocuments);
        PAL::registerNotifyCallback("com.apple.WebKit.showLayerTree"_s, printLayerTreeForLiveDocuments);
        PAL::registerNotifyCallback("com.apple.WebKit.showGraphicsLayerTree"_s, printGraphicsLayerTreeForLiveDocuments);
        PAL::registerNotifyCallback("com.apple.WebKit.showPaintOrderTree"_s, printPaintOrderTreeForLiveDocuments);
        PAL::registerNotifyCallback("com.apple.WebKit.showLayoutTree"_s, Layout::printLayoutTreeForLiveDocuments);
#endif // ENABLE(TREE_DEBUGGING)

        PAL::registerNotifyCallback("com.apple.WebKit.showAllDocuments"_s, [] {
            unsigned numPages = 0;
            Page::forEachPage([&numPages](Page&) {
                ++numPages;
            });

            WTFLogAlways("%u live pages:", numPages);

            Page::forEachPage([](Page& page) {
                RefPtr localTopDocument = page.localTopDocument();
                if (!localTopDocument)
                    return;
                WTFLogAlways("Page %p with main document %p %s", &page, localTopDocument.get(), localTopDocument ? localTopDocument->url().string().utf8().data() : "");
            });

            WTFLogAlways("%u live documents:", Document::allDocuments().size());
            for (auto& document : Document::allDocuments()) {
                const char* documentType = is<SVGDocument>(document.get()) ? "SVGDocument" : "Document";
                WTFLogAlways("%s %p %" PRIu64 "-%s (refCount %d, referencingNodeCount %d) %s", documentType, document.ptr(), document->identifier().processIdentifier().toUInt64(), document->identifier().toString().utf8().data(), document->refCount(), document->referencingNodeCount(), document->url().string().utf8().data());
            }
        });
    });
}

void Page::addSchedulePair(Ref<SchedulePair>&& pair)
{
    if (!m_scheduledRunLoopPairs)
        m_scheduledRunLoopPairs = makeUnique<SchedulePairHashSet>();
    m_scheduledRunLoopPairs->add(pair.ptr());

    for (RefPtr frame = &m_mainFrame.get(); frame; frame = frame->tree().traverseNext()) {
        RefPtr localFrame = dynamicDowncast<LocalFrame>(frame);
        if (!localFrame)
            continue;
        if (RefPtr documentLoader = localFrame->loader().documentLoader())
            documentLoader->schedule(pair);
        if (RefPtr documentLoader = localFrame->loader().provisionalDocumentLoader())
            documentLoader->schedule(pair);
    }

    // FIXME: make SharedTimerMac use these SchedulePairs.
}

void Page::removeSchedulePair(Ref<SchedulePair>&& pair)
{
    ASSERT(m_scheduledRunLoopPairs);
    if (!m_scheduledRunLoopPairs)
        return;

    m_scheduledRunLoopPairs->remove(pair.ptr());

    for (RefPtr frame = &m_mainFrame.get(); frame; frame = frame->tree().traverseNext()) {
        RefPtr localFrame = dynamicDowncast<LocalFrame>(frame);
        if (!localFrame)
            continue;
        if (RefPtr documentLoader = localFrame->loader().documentLoader())
            documentLoader->unschedule(pair);
        if (RefPtr documentLoader = localFrame->loader().provisionalDocumentLoader())
            documentLoader->unschedule(pair);
    }
}

const String& Page::presentingApplicationBundleIdentifier() const
{
    return m_presentingApplicationBundleIdentifier;
}

void Page::setPresentingApplicationBundleIdentifier(String&& bundleIdentifier)
{
    m_presentingApplicationBundleIdentifier = WTFMove(bundleIdentifier);
    PlatformMediaSessionManager::updateNowPlayingInfoIfNecessary();
}

} // namespace WebCore
