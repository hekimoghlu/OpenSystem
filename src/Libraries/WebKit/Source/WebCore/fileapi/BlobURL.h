/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include <wtf/URL.h>

namespace WebCore {

class SecurityOrigin;

// Blob URLs are of the form
//     blob:%escaped_origin%/%UUID%
// For public urls, the origin of the host page is encoded in the URL value to
// allow easy lookup of the origin when security checks need to be performed.
// When loading blobs via ResourceHandle or when reading blobs via FileReader
// the loader conducts security checks that examine the origin of host page
// encoded in the public blob url. The origin baked into internal blob urls
// is a simple constant value, "blobinternal://", internal urls should not
// be used with ResourceHandle or FileReader.
class BlobURL {
public:
    static URL createPublicURL(SecurityOrigin*);
    static URL createInternalURL();

    static URL getOriginURL(const URL&);
    static bool isSecureBlobURL(const URL&);
#if ASSERT_ENABLED
    static bool isInternalURL(const URL&);
#endif

private:
    static URL createBlobURL(StringView originString);
    BlobURL() { }
};

} // namespace WebCore
