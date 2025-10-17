/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

#include "DataTransfer.h"
#include "ScriptWrappable.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class DOMFileSystem;
class DataTransferListItem;
class File;
class FileSystemEntry;
class ScriptExecutionContext;
class StringCallback;

class DataTransferItem : public RefCounted<DataTransferItem> {
public:
    static Ref<DataTransferItem> create(WeakPtr<DataTransferItemList>&&, const String&);
    static Ref<DataTransferItem> create(WeakPtr<DataTransferItemList>&&, const String&, Ref<File>&&);

    ~DataTransferItem();

    RefPtr<File> file() { return m_file; }
    void clearListAndPutIntoDisabledMode();

    bool isFile() const { return m_file; }
    String kind() const;
    String type() const;
    void getAsString(Document&, RefPtr<StringCallback>&&) const;
    RefPtr<File> getAsFile() const;
    RefPtr<FileSystemEntry> getAsEntry(ScriptExecutionContext&) const;

private:
    DataTransferItem(WeakPtr<DataTransferItemList>&&, const String&);
    DataTransferItem(WeakPtr<DataTransferItemList>&&, const String&, Ref<File>&&);

    bool isInDisabledMode() const { return !m_list; }

    WeakPtr<DataTransferItemList> m_list;
    const String m_type;
    RefPtr<File> m_file;
};

}
