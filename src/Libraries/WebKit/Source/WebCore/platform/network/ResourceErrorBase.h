/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ResourceError;

WEBCORE_EXPORT extern const ASCIILiteral errorDomainWebKitInternal; // Used for errors that won't be exposed to clients.
WEBCORE_EXPORT extern const ASCIILiteral errorDomainWebKitServiceWorker; // Used for errors that happen when loading a resource from a service worker.

enum class ResourceErrorBaseType : uint8_t {
    Null,
    General,
    AccessControl,
    Cancellation,
    Timeout
};

class ResourceErrorBase {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ResourceErrorBase, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT ResourceError isolatedCopy() const;

    const String& domain() const { lazyInit(); return m_domain; }
    int errorCode() const { lazyInit(); return m_errorCode; }
    const URL& failingURL() const { lazyInit(); return m_failingURL; }
    const String& localizedDescription() const { lazyInit(); return m_localizedDescription; }

    String sanitizedDescription() const { return m_isSanitized  == IsSanitized::Yes ? m_localizedDescription : "Load failed"_s; }

    using Type = ResourceErrorBaseType;

    enum class IsSanitized : bool { No, Yes };

    enum class ErrorRecoveryMethod : bool {
        NoRecovery,
        HTTPFallback
    };

    bool isNull() const { return m_type == Type::Null; }
    bool isGeneral() const { return m_type == Type::General; }
    bool isAccessControl() const { return m_type == Type::AccessControl; }
    bool isCancellation() const { return m_type == Type::Cancellation; }
    bool isTimeout() const { return m_type == Type::Timeout; }

    static bool compare(const ResourceError&, const ResourceError&);

    WEBCORE_EXPORT void setType(Type);
    Type type() const { return m_type; }

    bool isSanitized() const { return m_isSanitized == IsSanitized::Yes; }
    void setAsSanitized() { m_isSanitized = IsSanitized::Yes; }

protected:
    ResourceErrorBase(Type type) : m_type(type) { }

    ResourceErrorBase(const String& domain, int errorCode, const URL& failingURL, const String& localizedDescription, Type type, IsSanitized isSanitized)
        : m_domain(domain)
        , m_failingURL(failingURL)
        , m_localizedDescription(localizedDescription)
        , m_errorCode(errorCode)
        , m_type(type)
        , m_isSanitized(isSanitized)
    {
    }

    WEBCORE_EXPORT void lazyInit() const;

    // The ResourceError subclass may "shadow" this method to lazily initialize platform specific fields
    void platformLazyInit() {}

    // The ResourceError subclass may "shadow" this method to compare platform specific fields
    static bool platformCompare(const ResourceError&, const ResourceError&) { return true; }

    String m_domain;
    URL m_failingURL;
    String m_localizedDescription;
    int m_errorCode { 0 };
    Type m_type { Type::General };
    IsSanitized m_isSanitized { IsSanitized::No };

private:
    const ResourceError& asResourceError() const;
};

// FIXME: Replace this with std::source_location when libc++ on the macOS and iOS bots is new enough to support C++ 20.
// WEBCORE_EXPORT ResourceError internalError(const URL&, std::source_location = std::source_location::current());
WEBCORE_EXPORT ResourceError createInternalError(const URL&, ASCIILiteral filename, uint32_t line, ASCIILiteral functionName);
#define internalError(url) createInternalError(url, ASCIILiteral::fromLiteralUnsafe(__FILE__), __LINE__, ASCIILiteral::fromLiteralUnsafe(__FUNCTION__))

WEBCORE_EXPORT ResourceError badResponseHeadersError(const URL&);

inline bool operator==(const ResourceError& a, const ResourceError& b) { return ResourceErrorBase::compare(a, b); }

} // namespace WebCore
