/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#include "ServiceWorkerClientData.h"

#include "AdvancedPrivacyProtections.h"
#include "Document.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "SWClientConnection.h"
#include "WorkerGlobalScope.h"
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

static ServiceWorkerClientFrameType toServiceWorkerClientFrameType(ScriptExecutionContext& context)
{
    auto* document = dynamicDowncast<Document>(context);
    if (!document)
        return ServiceWorkerClientFrameType::None;

    auto* frame = document->frame();
    if (!frame)
        return ServiceWorkerClientFrameType::None;

    if (frame->isMainFrame()) {
        if (RefPtr window = document->domWindow()) {
            if (window->opener())
                return ServiceWorkerClientFrameType::Auxiliary;
        }
        return ServiceWorkerClientFrameType::TopLevel;
    }
    return ServiceWorkerClientFrameType::Nested;
}

ServiceWorkerClientData ServiceWorkerClientData::isolatedCopy() const &
{
    return { identifier, type, frameType, url.isolatedCopy(), ownerURL.isolatedCopy(), pageIdentifier, frameIdentifier, lastNavigationWasAppInitiated, advancedPrivacyProtections, isVisible, isFocused, focusOrder, crossThreadCopy(ancestorOrigins) };
}

ServiceWorkerClientData ServiceWorkerClientData::isolatedCopy() &&
{
    return { identifier, type, frameType, WTFMove(url).isolatedCopy(), WTFMove(ownerURL).isolatedCopy(), pageIdentifier, frameIdentifier, lastNavigationWasAppInitiated, advancedPrivacyProtections, isVisible, isFocused, focusOrder, crossThreadCopy(WTFMove(ancestorOrigins)) };
}

ServiceWorkerClientData ServiceWorkerClientData::from(ScriptExecutionContext& context)
{
    if (auto* document = dynamicDowncast<Document>(context)) {
        auto lastNavigationWasAppInitiated = document->loader() && document->loader()->lastNavigationWasAppInitiated() ? LastNavigationWasAppInitiated::Yes : LastNavigationWasAppInitiated::No;

        Vector<String> ancestorOrigins;
        if (auto* frame = document->frame()) {
            for (auto* ancestor = frame->tree().parent(); ancestor; ancestor = ancestor->tree().parent()) {
                if (auto* ancestorFrame = dynamicDowncast<LocalFrame>(ancestor))
                    ancestorOrigins.append(ancestorFrame->document()->securityOrigin().toString());
            }
        }

        return {
            context.identifier(),
            ServiceWorkerClientType::Window,
            toServiceWorkerClientFrameType(context),
            document->creationURL(),
            URL(),
            document->pageID(),
            document->frameID(),
            lastNavigationWasAppInitiated,
            context.advancedPrivacyProtections(),
            !document->hidden(),
            document->hasFocus(),
            0,
            WTFMove(ancestorOrigins)
        };
    }

    RELEASE_ASSERT(is<WorkerGlobalScope>(context));
    auto& scope = downcast<WorkerGlobalScope>(context);
    return {
        scope.identifier(),
        scope.type() == WebCore::WorkerGlobalScope::Type::SharedWorker ? ServiceWorkerClientType::Sharedworker : ServiceWorkerClientType::Worker,
        ServiceWorkerClientFrameType::None,
        scope.url(),
        scope.ownerURL(),
        { },
        { },
        LastNavigationWasAppInitiated::No,
        context.advancedPrivacyProtections(),
        false,
        false,
        0,
        { }
    };
}

} // namespace WebCore
