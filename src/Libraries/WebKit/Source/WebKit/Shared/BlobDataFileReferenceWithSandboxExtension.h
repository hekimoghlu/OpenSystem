/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#ifndef BlobDataFileReferenceWithSandboxExtension_h
#define BlobDataFileReferenceWithSandboxExtension_h

#include <WebCore/BlobDataFileReference.h>

namespace WebKit {

class SandboxExtension;

class BlobDataFileReferenceWithSandboxExtension final : public WebCore::BlobDataFileReference {
public:
    static Ref<BlobDataFileReference> create(const String& path, const String& replacementPath = { }, RefPtr<SandboxExtension>&& sandboxExtension = nullptr)
    {
        return adoptRef(*new BlobDataFileReferenceWithSandboxExtension(path, replacementPath, WTFMove(sandboxExtension)));
    }

private:
    BlobDataFileReferenceWithSandboxExtension(const String& path, const String& replacementPath, RefPtr<SandboxExtension>&&);
    virtual ~BlobDataFileReferenceWithSandboxExtension();

    void prepareForFileAccess() override;
    void revokeFileAccess() override;

    RefPtr<SandboxExtension> m_sandboxExtension;
};

}

#endif // BlobDataFileReferenceWithSandboxExtension_h
