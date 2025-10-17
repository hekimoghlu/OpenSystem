/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#include "BlobURL.h"

#include "DocumentInlines.h"
#include "SecurityOrigin.h"
#include "ThreadableBlobRegistry.h"
#include <wtf/URL.h>
#include <wtf/UUID.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

URL BlobURL::createPublicURL(SecurityOrigin* securityOrigin)
{
    ASSERT(securityOrigin);
    return createBlobURL(securityOrigin->toString());
}

URL BlobURL::createInternalURL()
{
    return createBlobURL("blobinternal://"_s);
}

static const Document* blobOwner(const SecurityOrigin& blobOrigin)
{
    if (!isMainThread())
        return nullptr;

    for (auto& document : Document::allDocuments()) {
        if (document->protectedSecurityOrigin()->isSameOriginAs(blobOrigin))
            return document.ptr();
    }
    return nullptr;
}

URL BlobURL::getOriginURL(const URL& url)
{
    ASSERT(url.protocolIsBlob());

    return URL(SecurityOrigin::createForBlobURL(url)->toString());
}

bool BlobURL::isSecureBlobURL(const URL& url)
{
    ASSERT(url.protocolIsBlob());

    // As per https://github.com/w3c/webappsec-mixed-content/issues/41, Blob URL is secure if the document that created it is secure.
    if (auto origin = ThreadableBlobRegistry::getCachedOrigin(url)) {
        if (auto* document = blobOwner(*origin))
            return document->isSecureContext();
    }
    return SecurityOrigin::isSecure(getOriginURL(url));
}

URL BlobURL::createBlobURL(StringView originString)
{
    ASSERT(!originString.isEmpty());
    String urlString = makeString("blob:"_s, originString, '/', WTF::UUID::createVersion4());
    return URL({ }, urlString);
}

#if ASSERT_ENABLED
bool BlobURL::isInternalURL(const URL& url)
{
    return url.string().startsWith("blob:blobinternal://"_s);
}
#endif

} // namespace WebCore
