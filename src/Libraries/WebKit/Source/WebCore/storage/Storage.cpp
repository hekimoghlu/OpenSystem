/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#include "Storage.h"

#include "Document.h"
#include "LegacySchemeRegistry.h"
#include "LocalFrame.h"
#include "Page.h"
#include "ScriptTelemetryCategory.h"
#include "SecurityOrigin.h"
#include "StorageArea.h"
#include "StorageType.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Storage);

Ref<Storage> Storage::create(LocalDOMWindow& window, Ref<StorageArea>&& storageArea)
{
    return adoptRef(*new Storage(window, WTFMove(storageArea)));
}

Storage::Storage(LocalDOMWindow& window, Ref<StorageArea>&& storageArea)
    : LocalDOMWindowProperty(&window)
    , m_storageArea(WTFMove(storageArea))
{
    ASSERT(frame());

    m_storageArea->incrementAccessCount();
}

Storage::~Storage()
{
    m_storageArea->decrementAccessCount();
}

unsigned Storage::length() const
{
    if (requiresScriptExecutionTelemetry())
        return 0;

    return m_storageArea->length();
}

String Storage::key(unsigned index) const
{
    if (requiresScriptExecutionTelemetry())
        return { };

    return m_storageArea->key(index);
}

String Storage::getItem(const String& key) const
{
    if (requiresScriptExecutionTelemetry())
        return { };

    return m_storageArea->item(key);
}

ExceptionOr<void> Storage::setItem(const String& key, const String& value)
{
    auto* frame = this->frame();
    if (!frame)
        return Exception { ExceptionCode::InvalidAccessError };

    if (requiresScriptExecutionTelemetry())
        return { };

    bool quotaException = false;
    m_storageArea->setItem(*frame, key, value, quotaException);
    if (quotaException)
        return Exception { ExceptionCode::QuotaExceededError };
    return { };
}

ExceptionOr<void> Storage::removeItem(const String& key)
{
    auto* frame = this->frame();
    if (!frame)
        return Exception { ExceptionCode::InvalidAccessError };

    if (requiresScriptExecutionTelemetry())
        return { };

    m_storageArea->removeItem(*frame, key);
    return { };
}

ExceptionOr<void> Storage::clear()
{
    auto* frame = this->frame();
    if (!frame)
        return Exception { ExceptionCode::InvalidAccessError };

    m_storageArea->clear(*frame);
    return { };
}

bool Storage::contains(const String& key) const
{
    return m_storageArea->contains(key);
}

bool Storage::isSupportedPropertyName(const String& propertyName) const
{
    return m_storageArea->contains(propertyName);
}

Vector<AtomString> Storage::supportedPropertyNames() const
{
    unsigned length = m_storageArea->length();
    return Vector<AtomString>(length, [this](size_t i) {
        return m_storageArea->key(i);
    });
}

Ref<StorageArea> Storage::protectedArea() const
{
    return m_storageArea;
}

bool Storage::requiresScriptExecutionTelemetry() const
{
    RefPtr document = window() ? window()->document() : nullptr;
    return document && document->requiresScriptExecutionTelemetry(ScriptTelemetryCategory::LocalStorage);
}

} // namespace WebCore
