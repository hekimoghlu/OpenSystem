/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#include "ResourceLoader.h"
#include "ResourceResponse.h"
#include "SharedBuffer.h"

namespace WebCore {

class SubstituteResource : public RefCounted<SubstituteResource> {
public:
    virtual ~SubstituteResource() = default;

    const URL& url() const { return m_url; }
    const ResourceResponse& response() const { return m_response; }
    FragmentedSharedBuffer& data() const { return *m_data.get(); }
    void append(const SharedBuffer& buffer) { m_data.append(buffer); }
    void clear() { m_data.empty(); }

    virtual void deliver(ResourceLoader& loader) { loader.deliverResponseAndData(m_response, m_data.copy()); }

protected:
    SubstituteResource(URL&& url, ResourceResponse&& response, Ref<FragmentedSharedBuffer>&& data)
        : m_url(WTFMove(url))
        , m_response(WTFMove(response))
        , m_data(WTFMove(data))
    {
    }

private:
    URL m_url;
    ResourceResponse m_response;
    SharedBufferBuilder m_data;
};

} // namespace WebCore
