/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#include "ResourceLoadNotifier.h"

#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "InspectorInstrumentation.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Page.h"
#include "ProgressTracker.h"
#include "ResourceLoader.h"
#include "SharedBuffer.h"

#if USE(QUICK_LOOK)
#include "QuickLook.h"
#endif

namespace WebCore {

ResourceLoadNotifier::ResourceLoadNotifier(LocalFrame& frame)
    : m_frame(frame)
{
}

Ref<LocalFrame> ResourceLoadNotifier::protectedFrame() const
{
    return m_frame.get();
}

void ResourceLoadNotifier::didReceiveAuthenticationChallenge(ResourceLoaderIdentifier identifier, DocumentLoader* loader, const AuthenticationChallenge& currentWebChallenge)
{
    protectedFrame()->protectedLoader()->client().dispatchDidReceiveAuthenticationChallenge(loader, identifier, currentWebChallenge);
}

void ResourceLoadNotifier::willSendRequest(ResourceLoader& loader, ResourceLoaderIdentifier identifier, ResourceRequest& clientRequest, const ResourceResponse& redirectResponse)
{
    protectedFrame()->protectedLoader()->applyUserAgentIfNeeded(clientRequest);

    dispatchWillSendRequest(loader.protectedDocumentLoader().get(), identifier, clientRequest, redirectResponse, loader.protectedCachedResource().get(), &loader);
}

void ResourceLoadNotifier::didReceiveResponse(ResourceLoader& loader, ResourceLoaderIdentifier identifier, const ResourceResponse& r)
{
    loader.documentLoader()->addResponse(r);

    if (RefPtr page = m_frame->page())
        page->checkedProgress()->incrementProgress(identifier, r);

    dispatchDidReceiveResponse(loader.protectedDocumentLoader().get(), identifier, r, &loader);
}

void ResourceLoadNotifier::didReceiveData(ResourceLoader& loader, ResourceLoaderIdentifier identifier, const SharedBuffer& buffer, int encodedDataLength)
{
    if (RefPtr page = m_frame->page())
        page->checkedProgress()->incrementProgress(identifier, buffer.size());

    dispatchDidReceiveData(loader.protectedDocumentLoader().get(), identifier, &buffer, buffer.size(), encodedDataLength);
}

void ResourceLoadNotifier::didFinishLoad(ResourceLoader& loader, ResourceLoaderIdentifier identifier, const NetworkLoadMetrics& networkLoadMetrics)
{    
    if (RefPtr page = m_frame->page())
        page->checkedProgress()->completeProgress(identifier);

    dispatchDidFinishLoading(loader.protectedDocumentLoader().get(), loader.options().mode == FetchOptions::Mode::Navigate ? IsMainResourceLoad::Yes : loader.options().mode == FetchOptions::Mode::Navigate ? IsMainResourceLoad::Yes : IsMainResourceLoad::No, identifier, networkLoadMetrics, &loader);
}

void ResourceLoadNotifier::didFailToLoad(ResourceLoader& loader, ResourceLoaderIdentifier identifier, const ResourceError& error)
{
    if (RefPtr page = m_frame->page())
        page->checkedProgress()->completeProgress(identifier);

    // Notifying the LocalFrameLoaderClient may cause the frame to be destroyed.
    Ref frame = m_frame.get();
    if (!error.isNull())
        frame->protectedLoader()->client().dispatchDidFailLoading(loader.protectedDocumentLoader().get(), loader.options().mode == FetchOptions::Mode::Navigate ? IsMainResourceLoad::Yes : IsMainResourceLoad::No, identifier, error);

    InspectorInstrumentation::didFailLoading(frame.ptr(), loader.protectedDocumentLoader().get(), identifier, error);
}

void ResourceLoadNotifier::assignIdentifierToInitialRequest(ResourceLoaderIdentifier identifier, IsMainResourceLoad isMainResourceLoad, DocumentLoader* loader, const ResourceRequest& request)
{
    bool pageIsProvisionallyLoading = false;
    if (RefPtr frameLoader = loader ? loader->frameLoader() : nullptr)
        pageIsProvisionallyLoading = frameLoader->provisionalDocumentLoader() == loader;

    if (pageIsProvisionallyLoading)
        m_initialRequestIdentifier = identifier;

    protectedFrame()->protectedLoader()->client().assignIdentifierToInitialRequest(identifier, isMainResourceLoad, loader, request);
}

void ResourceLoadNotifier::dispatchWillSendRequest(DocumentLoader* loader, ResourceLoaderIdentifier identifier, ResourceRequest& request, const ResourceResponse& redirectResponse, const CachedResource* cachedResource, ResourceLoader* resourceLoader)
{
#if USE(QUICK_LOOK)
    // Always allow QuickLook-generated URLs based on the protocol scheme.
    if (!request.isNull() && isQuickLookPreviewURL(request.url()))
        return;
#endif

    String oldRequestURL = request.url().string();

    Ref frame = m_frame.get();
    ASSERT(frame->loader().documentLoader());
    if (RefPtr documentLoader = m_frame->loader().documentLoader())
        documentLoader->didTellClientAboutLoad(request.url().string());

    frame->protectedLoader()->client().dispatchWillSendRequest(loader, identifier, request, redirectResponse);

    // If the URL changed, then we want to put that new URL in the "did tell client" set too.
    if (!request.isNull() && oldRequestURL != request.url().string()) {
        if (RefPtr documentLoader = m_frame->loader().documentLoader())
            documentLoader->didTellClientAboutLoad(request.url().string());
    }

    InspectorInstrumentation::willSendRequest(frame.ptr(), identifier, loader, request, redirectResponse, cachedResource, resourceLoader);
}

void ResourceLoadNotifier::dispatchDidReceiveResponse(DocumentLoader* loader, ResourceLoaderIdentifier identifier, const ResourceResponse& r, ResourceLoader* resourceLoader)
{
    // Notifying the LocalFrameLoaderClient may cause the frame to be destroyed.
    Ref frame = m_frame.get();
    frame->protectedLoader()->client().dispatchDidReceiveResponse(loader, identifier, r);

    InspectorInstrumentation::didReceiveResourceResponse(frame, identifier, loader, r, resourceLoader);
}

void ResourceLoadNotifier::dispatchDidReceiveData(DocumentLoader* loader, ResourceLoaderIdentifier identifier, const SharedBuffer* buffer, int expectedDataLength, int encodedDataLength)
{
    // Notifying the LocalFrameLoaderClient may cause the frame to be destroyed.
    Ref frame = m_frame.get();
    frame->protectedLoader()->client().dispatchDidReceiveContentLength(loader, identifier, expectedDataLength);

    InspectorInstrumentation::didReceiveData(frame.ptr(), identifier, buffer, encodedDataLength);
}

void ResourceLoadNotifier::dispatchDidFinishLoading(DocumentLoader* loader, IsMainResourceLoad isMainResourceLoad, ResourceLoaderIdentifier identifier, const NetworkLoadMetrics& networkLoadMetrics, ResourceLoader* resourceLoader)
{
    // Notifying the LocalFrameLoaderClient may cause the frame to be destroyed.
    Ref frame = m_frame.get();
    frame->protectedLoader()->client().dispatchDidFinishLoading(loader, isMainResourceLoad, identifier);

    InspectorInstrumentation::didFinishLoading(frame.ptr(), loader, identifier, networkLoadMetrics, resourceLoader);
}

void ResourceLoadNotifier::dispatchDidFailLoading(DocumentLoader* loader, IsMainResourceLoad isMainResourceLoad, ResourceLoaderIdentifier identifier, const ResourceError& error)
{
    // Notifying the LocalFrameLoaderClient may cause the frame to be destroyed.
    Ref frame = m_frame.get();
    frame->protectedLoader()->client().dispatchDidFailLoading(loader, isMainResourceLoad, identifier, error);

    InspectorInstrumentation::didFailLoading(frame.ptr(), loader, identifier, error);
}

void ResourceLoadNotifier::sendRemainingDelegateMessages(DocumentLoader* loader, IsMainResourceLoad isMainResourceLoad, ResourceLoaderIdentifier identifier, const ResourceRequest& request, const ResourceResponse& response, const SharedBuffer* buffer, int expectedDataLength, int encodedDataLength, const ResourceError& error)
{
    // If the request is null, willSendRequest cancelled the load. We should
    // only dispatch didFailLoading in this case.
    if (request.isNull()) {
        ASSERT(error.isCancellation() || error.isAccessControl());
        dispatchDidFailLoading(loader, isMainResourceLoad, identifier, error);
        return;
    }

    if (!response.isNull())
        dispatchDidReceiveResponse(loader, identifier, response);

    if (expectedDataLength > 0)
        dispatchDidReceiveData(loader, identifier, buffer, expectedDataLength, encodedDataLength);

    if (error.isNull()) {
        NetworkLoadMetrics emptyMetrics;
        dispatchDidFinishLoading(loader, isMainResourceLoad, identifier, emptyMetrics, nullptr);
    } else
        dispatchDidFailLoading(loader, isMainResourceLoad, identifier, error);
}

} // namespace WebCore
