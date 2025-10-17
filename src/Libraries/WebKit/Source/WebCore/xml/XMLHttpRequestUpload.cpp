/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
#include "XMLHttpRequestUpload.h"

#include "ContextDestructionObserverInlines.h"
#include "EventNames.h"
#include "WebCoreOpaqueRoot.h"
#include "XMLHttpRequestProgressEvent.h"
#include <wtf/Assertions.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(XMLHttpRequestUpload);

XMLHttpRequestUpload::XMLHttpRequestUpload(XMLHttpRequest& request)
    : m_request(request)
{
}

void XMLHttpRequestUpload::eventListenersDidChange()
{
    m_request.updateHasRelevantEventListener();
}

bool XMLHttpRequestUpload::hasRelevantEventListener() const
{
    return hasEventListeners(eventNames().abortEvent)
        || hasEventListeners(eventNames().errorEvent)
        || hasEventListeners(eventNames().loadEvent)
        || hasEventListeners(eventNames().loadendEvent)
        || hasEventListeners(eventNames().loadstartEvent)
        || hasEventListeners(eventNames().progressEvent)
        || hasEventListeners(eventNames().timeoutEvent);
}

void XMLHttpRequestUpload::dispatchProgressEvent(const AtomString& type, unsigned long long loaded, unsigned long long total)
{
    ASSERT(type == eventNames().loadstartEvent || type == eventNames().progressEvent || type == eventNames().loadEvent || type == eventNames().loadendEvent || type == eventNames().abortEvent || type == eventNames().errorEvent || type == eventNames().timeoutEvent);

    // https://xhr.spec.whatwg.org/#firing-events-using-the-progressevent-interface
    dispatchEvent(XMLHttpRequestProgressEvent::create(type, !!total, loaded, total));
}

WebCoreOpaqueRoot root(XMLHttpRequestUpload* upload)
{
    return WebCoreOpaqueRoot { upload };
}

} // namespace WebCore
