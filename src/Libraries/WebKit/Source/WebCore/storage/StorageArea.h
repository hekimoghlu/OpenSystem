/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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

#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class LocalFrame;
class SecurityOrigin;
class StorageSyncManager;

enum class StorageType : uint8_t;

class SecurityOriginData;

class StorageArea : public RefCounted<StorageArea> {
public:
    virtual ~StorageArea() = default;

    virtual unsigned length() = 0;
    virtual String key(unsigned index) = 0;
    virtual String item(const String& key) = 0;
    virtual void setItem(LocalFrame& sourceFrame, const String& key, const String& value, bool& quotaException) = 0;
    virtual void removeItem(LocalFrame& sourceFrame, const String& key) = 0;
    virtual void clear(LocalFrame& sourceFrame) = 0;
    virtual bool contains(const String& key) = 0;

    virtual StorageType storageType() const = 0;

    virtual size_t memoryBytesUsedByCache() = 0;

    virtual void incrementAccessCount() { }
    virtual void decrementAccessCount() { }
    virtual void closeDatabaseIfIdle() { }
    virtual void prewarm() { }
};

} // namespace WebCore
