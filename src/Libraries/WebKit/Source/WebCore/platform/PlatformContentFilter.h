/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#ifndef PlatformContentFilter_h
#define PlatformContentFilter_h

#include "SharedBuffer.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class PlatformContentFilter;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PlatformContentFilter> : std::true_type { };
}

namespace WebCore {

class ContentFilterUnblockHandler;
class FragmentedSharedBuffer;
class ResourceRequest;
class ResourceResponse;
class SharedBuffer;

class PlatformContentFilter : public CanMakeWeakPtr<PlatformContentFilter> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PlatformContentFilter);
    WTF_MAKE_NONCOPYABLE(PlatformContentFilter);

public:
    enum class State {
        Stopped,
        Filtering,
        Allowed,
        Blocked,
    };

    bool needsMoreData() const { return m_state == State::Filtering; }
    bool didBlockData() const { return m_state == State::Blocked; }

    virtual ~PlatformContentFilter() = default;
    virtual void willSendRequest(ResourceRequest&, const ResourceResponse&) = 0;
    virtual void responseReceived(const ResourceResponse&) = 0;
    virtual void addData(const SharedBuffer&) = 0;
    virtual void finishedAddingData() = 0;
    virtual Ref<FragmentedSharedBuffer> replacementData() const = 0;
#if ENABLE(CONTENT_FILTERING)
    virtual ContentFilterUnblockHandler unblockHandler() const = 0;
#endif
    virtual String unblockRequestDeniedScript() const { return emptyString(); }

#if HAVE(AUDIT_TOKEN)
    const std::optional<audit_token_t> hostProcessAuditToken() const { return m_hostProcessAuditToken; }
    void setHostProcessAuditToken(const std::optional<audit_token_t>& token) { m_hostProcessAuditToken = token; }
#endif

protected:
    PlatformContentFilter() = default;
#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> m_hostProcessAuditToken;
#endif
    State m_state { State::Filtering };
};

} // namespace WebCore

#endif // PlatformContentFilter_h
