/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#include "ExceptionOr.h"
#include "LocalDOMWindowProperty.h"
#include "ScriptWrappable.h"

namespace WebCore {

class StorageArea;

class Storage final : public ScriptWrappable, public RefCounted<Storage>, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Storage);
public:
    static Ref<Storage> create(LocalDOMWindow&, Ref<StorageArea>&&);
    ~Storage();

    unsigned length() const;
    String key(unsigned index) const;
    String getItem(const String& key) const;
    ExceptionOr<void> setItem(const String& key, const String& value);
    ExceptionOr<void> removeItem(const String& key);
    ExceptionOr<void> clear();
    bool contains(const String& key) const;

    // Bindings support functions.
    bool isSupportedPropertyName(const String&) const;
    Vector<AtomString> supportedPropertyNames() const;

    StorageArea& area() const { return m_storageArea.get(); }
    Ref<StorageArea> protectedArea() const;

private:
    Storage(LocalDOMWindow&, Ref<StorageArea>&&);

    bool requiresScriptExecutionTelemetry() const;

    const Ref<StorageArea> m_storageArea;
};

} // namespace WebCore
