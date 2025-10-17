/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include "ArchiveResource.h"
#include <wtf/HashSet.h>

namespace WebCore {

class Archive : public RefCounted<Archive> {
public:
    virtual ~Archive();

    virtual bool shouldLoadFromArchiveOnly() const = 0;
    virtual bool shouldOverrideBaseURL() const = 0;
    virtual bool shouldUseMainResourceEncoding() const = 0;
    virtual bool shouldUseMainResourceURL() const = 0;

    ArchiveResource* mainResource() { return m_mainResource.get(); }
    const Vector<Ref<ArchiveResource>>& subresources() const { return m_subresources; }
    const Vector<Ref<Archive>>& subframeArchives() const { return m_subframeArchives; }
    WEBCORE_EXPORT Expected<Vector<String>, ArchiveError> saveResourcesToDisk(const String& directory);

protected:
    // These methods are meant for subclasses for different archive types to add resources in to the archive,
    // and should not be exposed as archives should be immutable to clients
    void setMainResource(Ref<ArchiveResource>&& mainResource) { m_mainResource = WTFMove(mainResource); }
    void addSubresource(Ref<ArchiveResource>&& resource) { m_subresources.append(WTFMove(resource)); }
    void addSubframeArchive(Ref<Archive>&& subframeArchive) { m_subframeArchives.append(WTFMove(subframeArchive)); }

    void clearAllSubframeArchives();

private:
    void clearAllSubframeArchives(UncheckedKeyHashSet<Archive*>&);

    RefPtr<ArchiveResource> m_mainResource;
    Vector<Ref<ArchiveResource>> m_subresources;
    Vector<Ref<Archive>> m_subframeArchives;
};

} // namespace WebCore
