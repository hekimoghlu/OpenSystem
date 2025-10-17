/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

#include <wtf/RefCounted.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WebCoreOpaqueRoot;

class WorkerLocation : public RefCounted<WorkerLocation> {
public:
    static Ref<WorkerLocation> create(URL&& url, String&& origin) { return adoptRef(*new WorkerLocation(WTFMove(url), WTFMove(origin))); }

    const URL& url() const { return m_url; }
    String href() const;

    // URI decomposition attributes
    String protocol() const;
    String host() const;
    String hostname() const;
    String port() const;
    String pathname() const;
    String search() const;
    String hash() const;
    String origin() const;

private:
    WorkerLocation(URL&& url, String&& origin)
        : m_url(WTFMove(url))
        , m_origin(WTFMove(origin))
    {
    }

    URL m_url;
    String m_origin;
};

WebCoreOpaqueRoot root(WorkerLocation*);

} // namespace WebCore
