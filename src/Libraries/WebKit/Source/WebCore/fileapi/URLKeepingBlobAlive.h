/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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

#include "SecurityOriginData.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/URL.h>

namespace WebCore {

// URL class which keeps the blob alive if the URL is a blob URL.
class URLKeepingBlobAlive {
public:
    URLKeepingBlobAlive() = default;
    URLKeepingBlobAlive(const URL&, const std::optional<SecurityOriginData>& = std::nullopt);
    WEBCORE_EXPORT ~URLKeepingBlobAlive();

    URLKeepingBlobAlive(URLKeepingBlobAlive&&) = default;
    URLKeepingBlobAlive& operator=(URLKeepingBlobAlive&&);

    URLKeepingBlobAlive(const URLKeepingBlobAlive&) = delete;
    URLKeepingBlobAlive& operator=(const URLKeepingBlobAlive&) = delete;

    operator const URL&() const { return m_url; }
    const URL& url() const { return m_url; }
    bool isEmpty() const { return m_url.isEmpty(); }
    std::optional<SecurityOriginData> topOrigin() const { return m_topOrigin; }

    void clear();

    // We do not introduce a && version since it might break the register/unregister balance.
    WEBCORE_EXPORT URLKeepingBlobAlive WARN_UNUSED_RETURN isolatedCopy() const;

private:
    void registerBlobURLHandleIfNecessary();
    void unregisterBlobURLHandleIfNecessary();

    URL m_url;
    Markable<SecurityOriginData, SecurityOriginDataMarkableTraits> m_topOrigin;
};

} // namespace WebCore
