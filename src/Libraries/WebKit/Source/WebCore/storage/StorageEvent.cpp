/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
#include "StorageEvent.h"

#include "Storage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(StorageEvent);

Ref<StorageEvent> StorageEvent::createForBindings()
{
    return adoptRef(*new StorageEvent);
}

StorageEvent::~StorageEvent() = default;

Ref<StorageEvent> StorageEvent::create(const AtomString& type, const String& key, const String& oldValue, const String& newValue, const String& url, Storage* storageArea)
{
    return adoptRef(*new StorageEvent(type, key, oldValue, newValue, url, storageArea));
}

Ref<StorageEvent> StorageEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new StorageEvent(type, initializer, isTrusted));
}

StorageEvent::StorageEvent()
    : Event(EventInterfaceType::StorageEvent)
{
}

StorageEvent::StorageEvent(const AtomString& type, const String& key, const String& oldValue, const String& newValue, const String& url, Storage* storageArea)
    : Event(EventInterfaceType::StorageEvent, type, CanBubble::No, IsCancelable::No)
    , m_key(key)
    , m_oldValue(oldValue)
    , m_newValue(newValue)
    , m_url(url)
    , m_storageArea(storageArea)
{
}

StorageEvent::StorageEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::StorageEvent, type, initializer, isTrusted)
    , m_key(initializer.key)
    , m_oldValue(initializer.oldValue)
    , m_newValue(initializer.newValue)
    , m_url(initializer.url)
    , m_storageArea(initializer.storageArea)
{
}

void StorageEvent::initStorageEvent(const AtomString& type, bool canBubble, bool cancelable, const String& key, const String& oldValue, const String& newValue, const String& url, Storage* storageArea)
{
    if (isBeingDispatched())
        return;

    initEvent(type, canBubble, cancelable);

    m_key = key;
    m_oldValue = oldValue;
    m_newValue = newValue;
    m_url = url;
    m_storageArea = storageArea;
}

} // namespace WebCore
