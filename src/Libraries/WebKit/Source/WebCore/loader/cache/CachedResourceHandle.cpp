/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
#include "CachedResourceHandle.h"

#include "CachedResource.h"

namespace WebCore {

CachedResourceHandleBase::CachedResourceHandleBase()
    : m_resource(nullptr)
{
}

CachedResourceHandleBase::CachedResourceHandleBase(CachedResource& resource)
    : m_resource(&resource)
{
    m_resource->registerHandle(this);
}

CachedResourceHandleBase::CachedResourceHandleBase(CachedResource* resource)
    : m_resource(resource)
{
    if (m_resource)
        m_resource->registerHandle(this);
}

CachedResourceHandleBase::CachedResourceHandleBase(const CachedResourceHandleBase& other)
    : m_resource(other.m_resource)
{
    if (m_resource)
        m_resource->registerHandle(this);
}

CachedResourceHandleBase::~CachedResourceHandleBase()
{
    if (m_resource)
        m_resource->unregisterHandle(this);
}

CachedResource* CachedResourceHandleBase::get() const
{
    return m_resource.get();
}

void CachedResourceHandleBase::setResource(CachedResource* resource)
{
    if (resource == m_resource)
        return;
    if (m_resource)
        m_resource->unregisterHandle(this);
    m_resource = resource;
    if (m_resource)
        m_resource->registerHandle(this);
}

}
