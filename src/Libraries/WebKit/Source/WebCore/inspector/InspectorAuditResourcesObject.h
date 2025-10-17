/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include "CachedFontClient.h"
#include "CachedImageClient.h"
#include "CachedRawResourceClient.h"
#include "CachedResourceClient.h"
#include "CachedSVGDocumentClient.h"
#include "CachedStyleSheetClient.h"
#include "ExceptionOr.h"
#include <JavaScriptCore/InspectorAuditAgent.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RobinHoodHashMap.h>

namespace WebCore {

class CachedResource;
class Document;

class InspectorAuditResourcesObject : public RefCounted<InspectorAuditResourcesObject> {
public:
    static Ref<InspectorAuditResourcesObject> create(Inspector::InspectorAuditAgent& auditAgent)
    {
        return adoptRef(*new InspectorAuditResourcesObject(auditAgent));
    }

    ~InspectorAuditResourcesObject();

    struct Resource {
        String id;
        String url;
        String mimeType;
    };

    struct ResourceContent {
        String data;
        bool base64Encoded;
    };

    ExceptionOr<Vector<Resource>> getResources(Document&);
    ExceptionOr<ResourceContent> getResourceContent(Document&, const String& id);

private:
    explicit InspectorAuditResourcesObject(Inspector::InspectorAuditAgent&);

    CachedResourceClient& clientForResource(const CachedResource&);

    Inspector::InspectorAuditAgent& m_auditAgent;

    class InspectorAuditCachedResourceClient : public CachedResourceClient { };
    InspectorAuditCachedResourceClient m_cachedResourceClient;

    class InspectorAuditCachedFontClient : public CachedFontClient { };
    InspectorAuditCachedFontClient m_cachedFontClient;

    class InspectorAuditCachedImageClient final : public CachedImageClient {
        WTF_MAKE_FAST_ALLOCATED;
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InspectorAuditCachedImageClient);
    };
    InspectorAuditCachedImageClient m_cachedImageClient;

    class InspectorAuditCachedRawResourceClient : public CachedRawResourceClient { };
    InspectorAuditCachedRawResourceClient m_cachedRawResourceClient;

    class InspectorAuditCachedSVGDocumentClient : public CachedSVGDocumentClient { };
    InspectorAuditCachedSVGDocumentClient m_cachedSVGDocumentClient;

    class InspectorAuditCachedStyleSheetClient : public CachedStyleSheetClient { };
    InspectorAuditCachedStyleSheetClient m_cachedStyleSheetClient;

    MemoryCompactRobinHoodHashMap<String, CachedResource*> m_resources;
};

} // namespace WebCore
