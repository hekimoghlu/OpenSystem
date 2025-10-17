/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUPtr.h"
#include "WebGPUQuerySet.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class QuerySetImpl final : public QuerySet {
    WTF_MAKE_TZONE_ALLOCATED(QuerySetImpl);
public:
    static Ref<QuerySetImpl> create(WebGPUPtr<WGPUQuerySet>&& querySet, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new QuerySetImpl(WTFMove(querySet), convertToBackingContext));
    }

    virtual ~QuerySetImpl();

private:
    friend class DowncastConvertToBackingContext;

    QuerySetImpl(WebGPUPtr<WGPUQuerySet>&&, ConvertToBackingContext&);

    QuerySetImpl(const QuerySetImpl&) = delete;
    QuerySetImpl(QuerySetImpl&&) = delete;
    QuerySetImpl& operator=(const QuerySetImpl&) = delete;
    QuerySetImpl& operator=(QuerySetImpl&&) = delete;

    WGPUQuerySet backing() const { return m_backing.get(); }

    void destroy() final;

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPUQuerySet> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
