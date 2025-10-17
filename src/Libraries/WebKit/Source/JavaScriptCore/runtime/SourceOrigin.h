/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#include "ScriptFetcher.h"
#include <wtf/URL.h>

namespace JSC {

class SourceOrigin {
public:
    explicit SourceOrigin(const URL& url)
        : m_url(url)
    {
    }

    explicit SourceOrigin(const URL& url, Ref<ScriptFetcher>&& fetcher)
        : m_url(url)
        , m_fetcher(WTFMove(fetcher))
    {
    }

    SourceOrigin() = default;

    const URL& url() const { return m_url; }
    const String& string() const { return m_url.string(); }
    bool isNull() const { return url().isNull(); }

    ScriptFetcher* fetcher() const { return m_fetcher.get(); }

private:
    URL m_url;
    RefPtr<ScriptFetcher> m_fetcher;
};

} // namespace JSC
