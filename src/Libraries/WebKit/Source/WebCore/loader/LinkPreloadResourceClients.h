/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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

#include "CachedCSSStyleSheet.h"
#include "CachedFont.h"
#include "CachedFontClient.h"
#include "CachedImage.h"
#include "CachedImageClient.h"
#include "CachedRawResource.h"
#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include "CachedScript.h"
#include "CachedStyleSheetClient.h"
#include "CachedTextTrack.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class LinkLoader;

class LinkPreloadResourceClient {
public:
    virtual ~LinkPreloadResourceClient() = default;

    void triggerEvents(const CachedResource&);

    virtual void clear() = 0;

protected:
    LinkPreloadResourceClient(LinkLoader&, CachedResource&);

    void addResource(CachedResourceClient& client)
    {
        m_resource->addClient(client);
    }

    void clearResource(CachedResourceClient& client)
    {
        if (!m_resource)
            return;

        m_resource->removeClient(client);
        m_resource = nullptr;
    }

    CachedResource* ownedResource() { return m_resource.get(); }

private:
    SingleThreadWeakPtr<LinkLoader> m_loader;
    CachedResourceHandle<CachedResource> m_resource;
};

class LinkPreloadDefaultResourceClient : public LinkPreloadResourceClient, CachedResourceClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    LinkPreloadDefaultResourceClient(LinkLoader& loader, CachedResource& resource)
        : LinkPreloadResourceClient(loader, resource)
    {
        addResource(*this);
    }

private:
    void notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final { triggerEvents(resource); }
    void clear() final { clearResource(*this); }
    bool shouldMarkAsReferenced() const final { return false; }
};

class LinkPreloadStyleResourceClient : public LinkPreloadResourceClient, public CachedStyleSheetClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    LinkPreloadStyleResourceClient(LinkLoader& loader, CachedCSSStyleSheet& resource)
        : LinkPreloadResourceClient(loader, resource)
    {
        addResource(*this);
    }

private:
    void setCSSStyleSheet(const String&, const URL&, ASCIILiteral, const CachedCSSStyleSheet* resource) final
    {
        ASSERT(resource);
        ASSERT(ownedResource() == resource);
        triggerEvents(*resource);
    }

    void clear() final { clearResource(*this); }
    bool shouldMarkAsReferenced() const final { return false; }
};

class LinkPreloadImageResourceClient : public LinkPreloadResourceClient, public CachedImageClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LinkPreloadImageResourceClient);
public:
    LinkPreloadImageResourceClient(LinkLoader& loader, CachedImage& resource)
        : LinkPreloadResourceClient(loader, resource)
    {
        addResource(*this);
    }

private:
    void notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final { triggerEvents(resource); }
    void clear() final { clearResource(*this); }
    bool shouldMarkAsReferenced() const final { return false; }
};

class LinkPreloadFontResourceClient : public LinkPreloadResourceClient, public CachedFontClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    LinkPreloadFontResourceClient(LinkLoader& loader, CachedFont& resource)
        : LinkPreloadResourceClient(loader, resource)
    {
        addResource(*this);
    }

private:
    void fontLoaded(CachedFont& resource) final
    {
        ASSERT(ownedResource() == &resource);
        triggerEvents(resource);
    }

    void clear() final { clearResource(*this); }
    bool shouldMarkAsReferenced() const final { return false; }
};

class LinkPreloadRawResourceClient : public LinkPreloadResourceClient, public CachedRawResourceClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    LinkPreloadRawResourceClient(LinkLoader& loader, CachedRawResource& resource)
        : LinkPreloadResourceClient(loader, resource)
    {
        addResource(*this);
    }

private:
    void notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final { triggerEvents(resource); }
    void clear() final { clearResource(*this); }
    bool shouldMarkAsReferenced() const final { return false; }
};

}
