/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#include "CachedSVGDocumentReference.h"

#include "CachedResourceHandle.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedResourceRequestInitiatorTypes.h"
#include "CachedSVGDocument.h"

namespace WebCore {

CachedSVGDocumentReference::CachedSVGDocumentReference(const String& url)
    : m_url(url)
{
}

CachedSVGDocumentReference::~CachedSVGDocumentReference()
{
    if (CachedResourceHandle document = m_document)
        document->removeClient(*this);
}

void CachedSVGDocumentReference::load(CachedResourceLoader& loader, const ResourceLoaderOptions& options)
{
    if (m_loadRequested)
        return;

    auto fetchOptions = options;
    fetchOptions.mode = FetchOptions::Mode::SameOrigin;
    CachedResourceRequest request(ResourceRequest(loader.document()->completeURL(m_url)), fetchOptions);
    request.setInitiatorType(cachedResourceRequestInitiatorTypes().css);
    m_document = loader.requestSVGDocument(WTFMove(request)).value_or(nullptr);
    if (CachedResourceHandle document = m_document)
        document->addClient(*this);

    m_loadRequested = true;
}

}
