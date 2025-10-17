/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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

#include "Archive.h"
#include "MarkupExclusionRule.h"
#include <wtf/Function.h>

namespace WebCore {

class LocalFrame;
class Node;

struct MarkupExclusionRule;
struct SimpleRange;

class LegacyWebArchive final : public Archive {
public:
    WEBCORE_EXPORT static Ref<LegacyWebArchive> create();
    WEBCORE_EXPORT static RefPtr<LegacyWebArchive> create(FragmentedSharedBuffer&);
    WEBCORE_EXPORT static RefPtr<LegacyWebArchive> create(const URL&, FragmentedSharedBuffer&);
    WEBCORE_EXPORT static Ref<LegacyWebArchive> create(Ref<ArchiveResource>&& mainResource, Vector<Ref<ArchiveResource>>&& subresources, Vector<Ref<LegacyWebArchive>>&& subframeArchives);
    WEBCORE_EXPORT static RefPtr<LegacyWebArchive> create(Node&, Function<bool(LocalFrame&)>&& frameFilter = { }, const Vector<MarkupExclusionRule>& markupExclusionRules = { }, const String& mainFrameFileName = { });
    WEBCORE_EXPORT static RefPtr<LegacyWebArchive> create(LocalFrame&);
    WEBCORE_EXPORT static RefPtr<LegacyWebArchive> createFromSelection(LocalFrame*);
    WEBCORE_EXPORT static RefPtr<LegacyWebArchive> create(const SimpleRange&);

    WEBCORE_EXPORT RetainPtr<CFDataRef> rawDataRepresentation();

private:
    LegacyWebArchive() = default;

    bool shouldLoadFromArchiveOnly() const final { return false; }
    bool shouldOverrideBaseURL() const final { return false; }
    bool shouldUseMainResourceEncoding() const final { return true; }
    bool shouldUseMainResourceURL() const final { return true; }

    enum MainResourceStatus { Subresource, MainResource };

    static RefPtr<LegacyWebArchive> create(const String& markupString, LocalFrame&, Vector<Ref<Node>>&& nodes, Function<bool(LocalFrame&)>&& frameFilter, const Vector<MarkupExclusionRule>& markupExclusionRules = { }, const String& mainResourceFileName = { });
    static RefPtr<ArchiveResource> createResource(CFDictionaryRef);
    static ResourceResponse createResourceResponseFromMacArchivedData(CFDataRef);
    static ResourceResponse createResourceResponseFromPropertyListData(CFDataRef, CFStringRef responseDataType);
    static RetainPtr<CFDataRef> createPropertyListRepresentation(const ResourceResponse&);
    static RetainPtr<CFDictionaryRef> createPropertyListRepresentation(Archive&);
    static RetainPtr<CFDictionaryRef> createPropertyListRepresentation(ArchiveResource*, MainResourceStatus);

    bool extract(CFDictionaryRef);
};

} // namespace WebCore
