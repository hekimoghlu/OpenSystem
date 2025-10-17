/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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

#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/MonotonicTime.h>
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

typedef struct _GInetAddress GInetAddress;

namespace WebKit {

class DNSCache : public RefCounted<DNSCache> {
public:
    static Ref<DNSCache> create();
    ~DNSCache() = default;

    enum class Type { Default, IPv4Only, IPv6Only };
    std::optional<Vector<GRefPtr<GInetAddress>>> lookup(const CString& host, Type = Type::Default);
    void update(const CString& host, Vector<GRefPtr<GInetAddress>>&&, Type = Type::Default);
    void clear();

private:
    DNSCache();

    struct CachedResponse {
        Vector<GRefPtr<GInetAddress>> addressList;
        MonotonicTime expirationTime;
    };

    using DNSCacheMap = HashMap<CString, CachedResponse>;

    DNSCacheMap& mapForType(Type) WTF_REQUIRES_LOCK(m_lock);
    void removeExpiredResponsesFired();
    void removeExpiredResponsesInMap(DNSCacheMap&);
    void pruneResponsesInMap(DNSCacheMap&);

    Lock m_lock;
    DNSCacheMap m_dnsMap WTF_GUARDED_BY_LOCK(m_lock);
    DNSCacheMap m_ipv4Map;
    DNSCacheMap m_ipv6Map;
    RunLoop::Timer m_expiredTimer;
};

} // namespace WebKit
