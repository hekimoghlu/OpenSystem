/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#include "SecurityOriginHash.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>

namespace PAL {
class SessionID;
}

namespace WebCore {

class Document;
class Page;
class SecurityOrigin;
class StorageArea;
class StorageNamespace;

class StorageNamespaceProvider : public RefCounted<StorageNamespaceProvider> {
public:
    WEBCORE_EXPORT StorageNamespaceProvider();
    WEBCORE_EXPORT virtual ~StorageNamespaceProvider();

    Ref<StorageArea> localStorageArea(Document&);
    Ref<StorageArea> sessionStorageArea(Document&);

    enum class ShouldCreateNamespace : bool { No, Yes };
    virtual RefPtr<StorageNamespace> sessionStorageNamespace(const SecurityOrigin&, Page&, ShouldCreateNamespace = ShouldCreateNamespace::Yes) = 0;

    WEBCORE_EXPORT void setSessionIDForTesting(PAL::SessionID);

    void setSessionStorageQuota(unsigned quota) { m_sessionStorageQuota = quota; }
    virtual void cloneSessionStorageNamespaceForPage(Page&, Page&) { RELEASE_ASSERT_NOT_REACHED(); }

protected:
    StorageNamespace* optionalLocalStorageNamespace() { return m_localStorageNamespace.get(); }
    unsigned sessionStorageQuota() const { return m_sessionStorageQuota; }

private:
    friend class Internals;
    WEBCORE_EXPORT StorageNamespace& localStorageNamespace(PAL::SessionID);
    StorageNamespace& transientLocalStorageNamespace(SecurityOrigin&, PAL::SessionID);

    virtual Ref<StorageNamespace> createLocalStorageNamespace(unsigned quota, PAL::SessionID) = 0;
    virtual Ref<StorageNamespace> createTransientLocalStorageNamespace(SecurityOrigin&, unsigned quota, PAL::SessionID) = 0;

    RefPtr<StorageNamespace> m_localStorageNamespace;
    HashMap<SecurityOriginData, RefPtr<StorageNamespace>> m_transientLocalStorageNamespaces;

    unsigned m_sessionStorageQuota { 0 };
};

} // namespace WebCore
