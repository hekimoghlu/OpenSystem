/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#include "PaymentSession.h"

#if ENABLE(APPLE_PAY)

#include "Document.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "PermissionsPolicy.h"
#include "SecurityOrigin.h"

namespace WebCore {

static bool isSecure(DocumentLoader& documentLoader)
{
    if (!documentLoader.response().url().protocolIs("https"_s))
        return false;

    if (!documentLoader.response().certificateInfo() || documentLoader.response().certificateInfo()->containsNonRootSHA1SignedCertificate())
        return false;

    return true;
}

ExceptionOr<void> PaymentSession::canCreateSession(Document& document)
{
    if (!PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Payment, document))
        return Exception { ExceptionCode::SecurityError, "Third-party iframes are not allowed to request payments unless explicitly allowed via Feature-Policy (payment)"_s };

    if (!document.frame())
        return Exception { ExceptionCode::InvalidAccessError, "Trying to start an Apple Pay session from an inactive document."_s };

    if (!isSecure(*document.loader()))
        return Exception { ExceptionCode::InvalidAccessError, "Trying to start an Apple Pay session from an insecure document."_s };

    RefPtr mainFrameDocument = document.protectedMainFrameDocument();
    if (!mainFrameDocument) {
        LOG_ONCE(SiteIsolation, "Unable to properly calculate PaymentSession::canCreateSession() without access to the main frame document ");
        return Exception { ExceptionCode::InvalidAccessError, "Trying to start an Apple Pay session from a site isolated iframe"_s };
    }

    if (!document.isTopDocument()) {
        for (RefPtr ancestorDocument = document.parentDocument(); ancestorDocument != mainFrameDocument.get(); ancestorDocument = ancestorDocument->parentDocument()) {
            if (!isSecure(*ancestorDocument->loader()))
                return Exception { ExceptionCode::InvalidAccessError, "Trying to start an Apple Pay session from a document with an insecure parent frame."_s };
        }
    }

    return { };
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
