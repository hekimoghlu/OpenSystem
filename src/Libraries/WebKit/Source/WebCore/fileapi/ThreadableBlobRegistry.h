/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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

#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class BlobPart;
class SecurityOrigin;
class SecurityOriginData;
class URLKeepingBlobAlive;

struct PolicyContainer;

class ThreadableBlobRegistry {
public:
    static void registerBlobURL(SecurityOrigin*, PolicyContainer&&, const URL&, const URL& srcURL, const std::optional<SecurityOriginData>& topOrigin);
    static void registerBlobURL(SecurityOrigin*, PolicyContainer&&, const URLKeepingBlobAlive&, const URL& srcURL);
    static void registerInternalFileBlobURL(const URL&, const String& path, const String& replacementPath, const String& contentType);
    static void registerInternalBlobURL(const URL&, Vector<BlobPart>&& blobParts, const String& contentType);
    static void registerInternalBlobURLOptionallyFileBacked(const URL&, const URL& srcURL, const String& fileBackedPath, const String& contentType);
    static void registerInternalBlobURLForSlice(const URL& newURL, const URL& srcURL, long long start, long long end, const String& contentType);
    static void unregisterBlobURL(const URL&, const std::optional<SecurityOriginData>& topOrigin);
    static void unregisterBlobURL(const URLKeepingBlobAlive&);

    static void registerBlobURLHandle(const URL&, const std::optional<SecurityOriginData>& topOrigin);
    static void unregisterBlobURLHandle(const URL&, const std::optional<SecurityOriginData>& topOrigin);

    WEBCORE_EXPORT static String blobType(const URL&);
    WEBCORE_EXPORT static unsigned long long blobSize(const URL&);

    // Returns the origin for the given blob URL. This is because we are not able to embed the unique security origin or the origin of file URL
    // in the blob URL.
    static RefPtr<SecurityOrigin> getCachedOrigin(const URL&);
};

} // namespace WebCore
