/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include "StorageAreaImpl.h"

#include "StorageAreaMap.h"
#include <WebCore/Document.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <WebCore/SecurityOriginData.h>
#include <WebCore/Settings.h>
#include <WebCore/StorageType.h>

namespace WebKit {
using namespace WebCore;

Ref<StorageAreaImpl> StorageAreaImpl::create(StorageAreaMap& storageAreaMap)
{
    return adoptRef(*new StorageAreaImpl(storageAreaMap));
}

StorageAreaImpl::StorageAreaImpl(StorageAreaMap& storageAreaMap)
    : m_storageAreaMap(storageAreaMap)
{
    storageAreaMap.incrementUseCount();
}

StorageAreaImpl::~StorageAreaImpl()
{
    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        storageAreaMap->decrementUseCount();
}

unsigned StorageAreaImpl::length()
{
    RefPtr storageAreaMap = m_storageAreaMap.get();
    return storageAreaMap ? storageAreaMap->length() : 0;
}

String StorageAreaImpl::key(unsigned index)
{
    RefPtr storageAreaMap = m_storageAreaMap.get();
    return storageAreaMap ? storageAreaMap->key(index) : nullString();
}

String StorageAreaImpl::item(const String& key)
{
    RefPtr storageAreaMap = m_storageAreaMap.get();
    return storageAreaMap ? storageAreaMap->item(key) : nullString();
}

void StorageAreaImpl::setItem(LocalFrame& sourceFrame, const String& key, const String& value, bool& quotaException)
{
    ASSERT(!value.isNull());

    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        storageAreaMap->setItem(sourceFrame, this, key, value, quotaException);
}

void StorageAreaImpl::removeItem(LocalFrame& sourceFrame, const String& key)
{
    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        storageAreaMap->removeItem(sourceFrame, this, key);
}

void StorageAreaImpl::clear(LocalFrame& sourceFrame)
{
    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        storageAreaMap->clear(sourceFrame, this);
}

bool StorageAreaImpl::contains(const String& key)
{
    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        return storageAreaMap->contains(key);

    return false;
}

StorageType StorageAreaImpl::storageType() const
{
    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        return storageAreaMap->type();

    // We probably need an Invalid type.
    return StorageType::Local;
}

size_t StorageAreaImpl::memoryBytesUsedByCache()
{
    return 0;
}

void StorageAreaImpl::prewarm()
{
    if (RefPtr storageAreaMap = m_storageAreaMap.get())
        storageAreaMap->connect();
}

} // namespace WebKit
