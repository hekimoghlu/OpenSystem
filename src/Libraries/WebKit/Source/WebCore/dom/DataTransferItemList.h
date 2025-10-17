/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "ContextDestructionObserver.h"
#include "DataTransfer.h"
#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DataTransferItem;
class Document;
class File;

class DataTransferItemList final : public ScriptWrappable, public ContextDestructionObserver, public CanMakeWeakPtr<DataTransferItemList> {
    WTF_MAKE_NONCOPYABLE(DataTransferItemList);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DataTransferItemList);
public:
    DataTransferItemList(Document&, DataTransfer&);
    ~DataTransferItemList();

    // DataTransfer owns DataTransferItemList, and DataTransfer is kept alive as long as DataTransferItemList is alive.
    void ref() { m_dataTransfer->ref(); }
    void deref() { m_dataTransfer->deref(); }
    DataTransfer& dataTransfer() { return m_dataTransfer.get(); }

    // DOM API
    unsigned length() const;
    RefPtr<DataTransferItem> item(unsigned index);
    bool isSupportedPropertyIndex(unsigned index);
    ExceptionOr<RefPtr<DataTransferItem>> add(Document&, const String& data, const String& type);
    RefPtr<DataTransferItem> add(Ref<File>&&);
    ExceptionOr<void> remove(unsigned index);
    void clear();

    void didClearStringData(const String& type);
    void didSetStringData(const String& type);
    bool hasItems() const { return m_items.has_value(); }
    const Vector<Ref<DataTransferItem>>& items() const
    {
        ASSERT(m_items);
        return *m_items;
    }

private:
    Vector<Ref<DataTransferItem>>& ensureItems() const;
    Document* document() const;

    WeakRef<DataTransfer> m_dataTransfer;
    mutable std::optional<Vector<Ref<DataTransferItem>>> m_items;
};

} // namespace WebCore
