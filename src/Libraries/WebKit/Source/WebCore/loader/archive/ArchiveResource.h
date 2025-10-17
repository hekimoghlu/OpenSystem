/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

#include "ArchiveError.h"
#include "SubstituteResource.h"

namespace WebCore {

class ArchiveResource : public SubstituteResource {
public:
    static RefPtr<ArchiveResource> create(RefPtr<FragmentedSharedBuffer>&&, const URL&, const ResourceResponse&);
    WEBCORE_EXPORT static RefPtr<ArchiveResource> create(RefPtr<FragmentedSharedBuffer>&&, const URL&, const String& mimeType, const String& textEncoding, const String& frameName, const ResourceResponse& = ResourceResponse(), const String& fileName = { });

    const String& mimeType() const { return m_mimeType; }
    const String& textEncoding() const { return m_textEncoding; }
    const String& frameName() const { return m_frameName; }
    const String& relativeFilePath() const { return m_relativeFilePath; }

    void ignoreWhenUnarchiving() { m_shouldIgnoreWhenUnarchiving = true; }
    bool shouldIgnoreWhenUnarchiving() const { return m_shouldIgnoreWhenUnarchiving; }
    void setRelativeFilePath(const String& relativeFilePath) { m_relativeFilePath = relativeFilePath; }
    Expected<String, ArchiveError> saveToDisk(const String& directory);

private:
    ArchiveResource(Ref<FragmentedSharedBuffer>&&, const URL&, const String& mimeType, const String& textEncoding, const String& frameName, const ResourceResponse&, const String& fileName = { });

    String m_mimeType;
    String m_textEncoding;
    String m_frameName;
    String m_relativeFilePath;

    bool m_shouldIgnoreWhenUnarchiving;
};

} // namespace WebCore
