/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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

#include "ResourceLoaderIdentifier.h"
#include <optional>
#include <wtf/Noncopyable.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class AuthenticationChallenge;
class CachedResource;
class DocumentLoader;
class LocalFrame;
class NetworkLoadMetrics;
class ResourceError;
class ResourceLoader;
class ResourceRequest;
class ResourceResponse;
class SharedBuffer;

enum class IsMainResourceLoad : bool;

class ResourceLoadNotifier {
    WTF_MAKE_NONCOPYABLE(ResourceLoadNotifier);
public:
    explicit ResourceLoadNotifier(LocalFrame&);

    void didReceiveAuthenticationChallenge(ResourceLoaderIdentifier, DocumentLoader*, const AuthenticationChallenge&);

    void willSendRequest(ResourceLoader&, ResourceLoaderIdentifier, ResourceRequest&, const ResourceResponse& redirectResponse);
    void didReceiveResponse(ResourceLoader&, ResourceLoaderIdentifier, const ResourceResponse&);
    void didReceiveData(ResourceLoader&, ResourceLoaderIdentifier, const SharedBuffer&, int encodedDataLength);
    void didFinishLoad(ResourceLoader&, ResourceLoaderIdentifier, const NetworkLoadMetrics&);
    void didFailToLoad(ResourceLoader&, ResourceLoaderIdentifier, const ResourceError&);

    void assignIdentifierToInitialRequest(ResourceLoaderIdentifier, IsMainResourceLoad, DocumentLoader*, const ResourceRequest&);
    void dispatchWillSendRequest(DocumentLoader*, ResourceLoaderIdentifier, ResourceRequest&, const ResourceResponse& redirectResponse, const CachedResource*, ResourceLoader* = nullptr);
    void dispatchDidReceiveResponse(DocumentLoader*, ResourceLoaderIdentifier, const ResourceResponse&, ResourceLoader* = nullptr);
    void dispatchDidReceiveData(DocumentLoader*, ResourceLoaderIdentifier, const SharedBuffer*, int expectedDataLength, int encodedDataLength);
    void dispatchDidFinishLoading(DocumentLoader*, IsMainResourceLoad, ResourceLoaderIdentifier, const NetworkLoadMetrics&, ResourceLoader*);
    void dispatchDidFailLoading(DocumentLoader*, IsMainResourceLoad, ResourceLoaderIdentifier, const ResourceError&);

    void sendRemainingDelegateMessages(DocumentLoader*, IsMainResourceLoad, ResourceLoaderIdentifier, const ResourceRequest&, const ResourceResponse&, const SharedBuffer*, int expectedDataLength, int encodedDataLength, const ResourceError&);

    bool isInitialRequestIdentifier(ResourceLoaderIdentifier identifier)
    {
        return m_initialRequestIdentifier == identifier;
    }

private:
    Ref<LocalFrame> protectedFrame() const;

    WeakRef<LocalFrame> m_frame;
    std::optional<ResourceLoaderIdentifier> m_initialRequestIdentifier;
};

} // namespace WebCore
