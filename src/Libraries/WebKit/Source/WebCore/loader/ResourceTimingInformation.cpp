/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
#include "ResourceTimingInformation.h"

#include "CachedResource.h"
#include "DocumentInlines.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "HTMLFrameOwnerElement.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "Performance.h"
#include "ResourceTiming.h"
#include "SecurityOrigin.h"

namespace WebCore {

bool ResourceTimingInformation::shouldAddResourceTiming(CachedResource& resource)
{
    return resource.resourceRequest().url().protocolIsInHTTPFamily()
        && !resource.loadFailedOrCanceled()
        && resource.options().loadedFromOpaqueSource == LoadedFromOpaqueSource::No;
}

void ResourceTimingInformation::addResourceTiming(CachedResource& resource, Document& document, ResourceTiming&& resourceTiming)
{
    if (!ResourceTimingInformation::shouldAddResourceTiming(resource))
        return;

    auto iterator = m_initiatorMap.find(resource);
    if (iterator == m_initiatorMap.end())
        return;

    InitiatorInfo& info = iterator->value;
    if (info.added == Added)
        return;

    RefPtr initiatorDocument = &document;
    if (resource.type() == CachedResource::Type::MainResource && document.frame() && document.frame()->loader().shouldReportResourceTimingToParentFrame()) {
        initiatorDocument = document.parentDocument();
        if (initiatorDocument)
            resourceTiming.updateExposure(initiatorDocument->protectedSecurityOrigin());
    }
    if (!initiatorDocument)
        return;

    RefPtr initiatorWindow = initiatorDocument->domWindow();
    if (!initiatorWindow)
        return;

    resourceTiming.overrideInitiatorType(info.type);

    initiatorWindow->protectedPerformance()->addResourceTiming(WTFMove(resourceTiming));

    info.added = Added;
}

void ResourceTimingInformation::removeResourceTiming(CachedResource& resource)
{
    m_initiatorMap.remove(resource);
}

void ResourceTimingInformation::storeResourceTimingInitiatorInformation(const CachedResourceHandle<CachedResource>& resource, const AtomString& initiatorType, LocalFrame* frame)
{
    ASSERT(resource.get());

    if (resource->type() == CachedResource::Type::MainResource) {
        // <iframe>s should report the initial navigation requested by the parent document, but not subsequent navigations.
        ASSERT(frame);
        if (frame->ownerElement()) {
            InitiatorInfo info = { frame->ownerElement()->localName(), NotYetAdded };
            m_initiatorMap.add(*resource, info);
        }
    } else {
        InitiatorInfo info = { initiatorType, NotYetAdded };
        m_initiatorMap.add(*resource, info);
    }
}

}
