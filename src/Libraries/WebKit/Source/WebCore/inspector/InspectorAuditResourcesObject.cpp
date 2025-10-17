/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include "InspectorAuditResourcesObject.h"

#include "CachedCSSStyleSheet.h"
#include "CachedFont.h"
#include "CachedImage.h"
#include "CachedRawResource.h"
#include "CachedResource.h"
#include "CachedSVGDocument.h"
#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "InspectorPageAgent.h"
#include <wtf/Vector.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using namespace Inspector;

#define ERROR_IF_NO_ACTIVE_AUDIT() \
    if (!m_auditAgent.hasActiveAudit()) \
        return Exception { ExceptionCode::NotAllowedError, "Cannot be called outside of a Web Inspector Audit"_s };

InspectorAuditResourcesObject::InspectorAuditResourcesObject(InspectorAuditAgent& auditAgent)
    : m_auditAgent(auditAgent)
{
}

InspectorAuditResourcesObject::~InspectorAuditResourcesObject()
{
    for (auto* cachedResource : m_resources.values())
        cachedResource->removeClient(clientForResource(*cachedResource));
}

ExceptionOr<Vector<InspectorAuditResourcesObject::Resource>> InspectorAuditResourcesObject::getResources(Document& document)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    Vector<Resource> resources;

    auto* frame = document.frame();
    if (!frame)
        return Exception { ExceptionCode::NotAllowedError, "Cannot be called with a detached document"_s };

    for (auto* cachedResource : InspectorPageAgent::cachedResourcesForFrame(frame)) {
        Resource resource;
        resource.url = cachedResource->url().string();
        resource.mimeType = cachedResource->mimeType();

        bool exists = false;
        for (const auto& entry : m_resources) {
            if (entry.value == cachedResource) {
                resource.id = entry.key;
                exists = true;
                break;
            }
        }
        if (!exists) {
            cachedResource->addClient(clientForResource(*cachedResource));

            resource.id = String::number(m_resources.size() + 1);
            m_resources.add(resource.id, cachedResource);
        }

        resources.append(WTFMove(resource));
    }

    return resources;
}

ExceptionOr<InspectorAuditResourcesObject::ResourceContent> InspectorAuditResourcesObject::getResourceContent(Document& document, const String& id)
{
    ERROR_IF_NO_ACTIVE_AUDIT();

    auto* frame = document.frame();
    if (!frame)
        return Exception { ExceptionCode::NotAllowedError, "Cannot be called with a detached document"_s };

    auto* cachedResource = m_resources.get(id);
    if (!cachedResource)
        return Exception { ExceptionCode::NotFoundError, makeString("Unknown identifier "_s, id) };

    Inspector::Protocol::ErrorString errorString;
    ResourceContent resourceContent;
    InspectorPageAgent::resourceContent(errorString, frame, cachedResource->url(), &resourceContent.data, &resourceContent.base64Encoded);
    if (!errorString.isEmpty())
        return Exception { ExceptionCode::NotFoundError, errorString };

    return resourceContent;
}

CachedResourceClient& InspectorAuditResourcesObject::clientForResource(const CachedResource& cachedResource)
{
    if (is<CachedCSSStyleSheet>(cachedResource))
        return m_cachedStyleSheetClient;

    if (is<CachedFont>(cachedResource))
        return m_cachedFontClient;

    if (is<CachedImage>(cachedResource))
        return m_cachedImageClient;

    if (is<CachedRawResource>(cachedResource))
        return m_cachedRawResourceClient;

    if (is<CachedSVGDocument>(cachedResource))
        return m_cachedSVGDocumentClient;

    return m_cachedResourceClient;
}

} // namespace WebCore
