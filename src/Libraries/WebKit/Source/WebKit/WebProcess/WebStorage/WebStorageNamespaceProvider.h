/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

#include "StorageNamespaceIdentifier.h"
#include "StorageNamespaceImpl.h"
#include <WebCore/StorageNamespaceProvider.h>

namespace WebKit {

class WebStorageNamespaceProvider final : public WebCore::StorageNamespaceProvider, public CanMakeWeakPtr<WebStorageNamespaceProvider> {
public:
    static Ref<WebStorageNamespaceProvider> getOrCreate();
    virtual ~WebStorageNamespaceProvider();

    static void incrementUseCount(const StorageNamespaceImpl::Identifier);
    static void decrementUseCount(const StorageNamespaceImpl::Identifier);

private:
    explicit WebStorageNamespaceProvider();

    Ref<WebCore::StorageNamespace> createLocalStorageNamespace(unsigned quota, PAL::SessionID) override;
    Ref<WebCore::StorageNamespace> createTransientLocalStorageNamespace(WebCore::SecurityOrigin&, unsigned quota, PAL::SessionID) override;

    RefPtr<WebCore::StorageNamespace> sessionStorageNamespace(const WebCore::SecurityOrigin&, WebCore::Page&, ShouldCreateNamespace) final;
    struct SessionStorageNamespaces {
        unsigned useCount { 0 };
        HashMap<WebCore::SecurityOriginData, RefPtr<WebCore::StorageNamespace>> map;
    };

    HashMap<StorageNamespaceImpl::Identifier, SessionStorageNamespaces> m_sessionStorageNamespaces;
};

}
