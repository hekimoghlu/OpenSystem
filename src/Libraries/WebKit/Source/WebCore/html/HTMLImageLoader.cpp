/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#include "HTMLImageLoader.h"

#include "CachedImage.h"
#include "CommonVM.h"
#include "Element.h"
#include "Event.h"
#include "EventNames.h"
#include "HTMLNames.h"
#include "HTMLObjectElement.h"
#include "HTMLVideoElement.h"
#include "LocalDOMWindow.h"
#include "Settings.h"

#include "JSDOMWindowBase.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSLock.h>

namespace WebCore {

HTMLImageLoader::HTMLImageLoader(Element& element)
    : ImageLoader(element)
{
}

HTMLImageLoader::~HTMLImageLoader() = default;

void HTMLImageLoader::dispatchLoadEvent()
{
#if ENABLE(VIDEO)
    // HTMLVideoElement uses this class to load the poster image, but it should not fire events for loading or failure.
    if (is<HTMLVideoElement>(element()))
        return;
#endif

#if PLATFORM(IOS_FAMILY)
    // iOS loads PDF inside <object> elements as images since we don't support loading them
    // as plugins (see logic in WebFrameLoaderClient::objectContentType()). However, WebKit
    // doesn't normally fire load/error events when loading <object> as plugins. Therefore,
    // firing such events for PDF loads on iOS can cause confusion on some sites.
    // See rdar://107795151.
    if (auto* objectElement = dynamicDowncast<HTMLObjectElement>(element())) {
        if (MIMETypeRegistry::isPDFMIMEType(objectElement->serviceType()))
            return;
    }
#endif

    bool errorOccurred = image()->errorOccurred();
    if (!errorOccurred && image()->response().httpStatusCode() >= 400)
        errorOccurred = is<HTMLObjectElement>(element()); // An <object> considers a 404 to be an error and should fire onerror.
    element().dispatchEvent(Event::create(errorOccurred ? eventNames().errorEvent : eventNames().loadEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

void HTMLImageLoader::notifyFinished(CachedResource&, const NetworkLoadMetrics& metrics, LoadWillContinueInAnotherProcess loadWillContinueInAnotherProcess)
{
    ASSERT(image());
    CachedImage& cachedImage = *image();

    Ref<Element> protect(element());
    ImageLoader::notifyFinished(cachedImage, metrics, loadWillContinueInAnotherProcess);

    bool loadError = cachedImage.errorOccurred() || cachedImage.response().httpStatusCode() >= 400;
    if (!loadError) {
        if (!element().isConnected()) {
            JSC::VM& vm = commonVM();
            JSC::JSLockHolder lock(vm);
            // FIXME: Adopt reportExtraMemoryVisited, and switch to reportExtraMemoryAllocated.
            // https://bugs.webkit.org/show_bug.cgi?id=142595
            vm.heap.deprecatedReportExtraMemory(cachedImage.encodedSize());
        }
    }

    if (loadError) {
        if (RefPtr objectElement = dynamicDowncast<HTMLObjectElement>(element()))
            objectElement->renderFallbackContent();
    }
}

}
