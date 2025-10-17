/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#include "ResourceError.h"

#include "LocalizedStrings.h"
#include "Logging.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ResourceErrorBase);

const ASCIILiteral errorDomainWebKitInternal = "WebKitInternal"_s;
const ASCIILiteral errorDomainWebKitServiceWorker = "WebKitServiceWorker"_s;

inline const ResourceError& ResourceErrorBase::asResourceError() const
{
    return *static_cast<const ResourceError*>(this);
}

ResourceError ResourceErrorBase::isolatedCopy() const
{
    lazyInit();

    ResourceError errorCopy;
    errorCopy.m_domain = m_domain.isolatedCopy();
    errorCopy.m_errorCode = m_errorCode;
    errorCopy.m_failingURL = m_failingURL.isolatedCopy();
    errorCopy.m_localizedDescription = m_localizedDescription.isolatedCopy();
    errorCopy.m_type = m_type;

    errorCopy.doPlatformIsolatedCopy(asResourceError());

    return errorCopy;
}

void ResourceErrorBase::lazyInit() const
{
    const_cast<ResourceError*>(static_cast<const ResourceError*>(this))->platformLazyInit();
}

void ResourceErrorBase::setType(Type type)
{
    // setType should only be used to specialize the error type.
    ASSERT(m_type == type || m_type == Type::General || m_type == Type::Null || (m_type == Type::Cancellation && type == Type::AccessControl));
    m_type = type;
}

bool ResourceErrorBase::compare(const ResourceError& a, const ResourceError& b)
{
    if (a.isNull() && b.isNull())
        return true;

    if (a.type() != b.type())
        return false;

    if (a.domain() != b.domain())
        return false;

    if (a.errorCode() != b.errorCode())
        return false;

    if (a.failingURL() != b.failingURL())
        return false;

    if (a.localizedDescription() != b.localizedDescription())
        return false;

    return ResourceError::platformCompare(a, b);
}

ResourceError createInternalError(const URL& url, ASCIILiteral filename, uint32_t line, ASCIILiteral functionName)
{
    // Always print internal errors to stderr so we have some chance to figure out what went wrong
    // when an internal error occurs unexpectedly. Release logging is insufficient because internal
    // errors occur unexpectedly and we don't want to require manual logging configuration in order
    // to record them.
    WTFReportError(filename.characters(), line, functionName.characters(), "WebKit encountered an internal error. This is a WebKit bug.");

    return ResourceError("WebKitErrorDomain"_s, 300, url, WEB_UI_STRING("WebKit encountered an internal error", "WebKitErrorInternal description"));
}

ResourceError badResponseHeadersError(const URL& url)
{
    return { errorDomainWebKitInternal, 0, url, "Response contained invalid HTTP headers"_s, ResourceError::Type::General };
}

} // namespace WebCore
