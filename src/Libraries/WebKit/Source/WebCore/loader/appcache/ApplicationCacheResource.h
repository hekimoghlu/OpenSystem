/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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

#include "SubstituteResource.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class ApplicationCacheResource : public SubstituteResource, public CanMakeWeakPtr<ApplicationCacheResource> {
public:
    enum Type {
        Master = 1 << 0,
        Manifest = 1 << 1,
        Explicit = 1 << 2,
        Foreign = 1 << 3,
        Fallback = 1 << 4
    };

    static Ref<ApplicationCacheResource> create(const URL&, const ResourceResponse&, unsigned type, RefPtr<FragmentedSharedBuffer>&& = SharedBuffer::create(), const String& path = String());

    unsigned type() const { return m_type; }
    void addType(unsigned type);
    
    void setStorageID(unsigned storageID) { m_storageID = storageID; }
    unsigned storageID() const { return m_storageID; }
    void clearStorageID() { m_storageID = 0; }
    int64_t estimatedSizeInStorage();

    const String& path() const { return m_path; }
    void setPath(const String& path) { m_path = path; }

#ifndef NDEBUG
    static void dumpType(unsigned type);
#endif
    
private:
    ApplicationCacheResource(URL&&, ResourceResponse&&, unsigned type, Ref<FragmentedSharedBuffer>&&, const String& path);

    void deliver(ResourceLoader&) override;

    unsigned m_type;
    unsigned m_storageID;
    int64_t m_estimatedSizeInStorage;
    String m_path;
};
    
} // namespace WebCore
