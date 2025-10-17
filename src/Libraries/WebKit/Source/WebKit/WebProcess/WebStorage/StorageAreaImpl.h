/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

#include "MessageReceiver.h"
#include "StorageAreaImplIdentifier.h"
#include <WebCore/StorageArea.h>
#include <wtf/HashMap.h>
#include <wtf/Identified.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

class StorageAreaMap;

class StorageAreaImpl final : public WebCore::StorageArea, public Identified<StorageAreaImplIdentifier> {
public:
    using Identifier = StorageAreaImplIdentifier;

    static Ref<StorageAreaImpl> create(StorageAreaMap&);
    virtual ~StorageAreaImpl();

private:
    StorageAreaImpl(StorageAreaMap&);

    // WebCore::StorageArea.
    unsigned length() override;
    String key(unsigned index) override;
    String item(const String& key) override;
    void setItem(WebCore::LocalFrame& sourceFrame, const String& key, const String& value, bool& quotaException) override;
    void removeItem(WebCore::LocalFrame& sourceFrame, const String& key) override;
    void clear(WebCore::LocalFrame& sourceFrame) override;
    bool contains(const String& key) override;
    WebCore::StorageType storageType() const override;
    size_t memoryBytesUsedByCache() override;
    void prewarm() final;

    WeakPtr<StorageAreaMap> m_storageAreaMap;
};

} // namespace WebKit
