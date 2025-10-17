/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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

#include "File.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class FileList final : public ScriptWrappable, public RefCounted<FileList> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(FileList, WEBCORE_EXPORT);
public:
    static Ref<FileList> create()
    {
        return adoptRef(*new FileList);
    }

    static Ref<FileList> create(Vector<Ref<File>>&& files)
    {
        return adoptRef(*new FileList(WTFMove(files)));
    }

    unsigned length() const { return m_files.size(); }
    WEBCORE_EXPORT File* item(unsigned index) const;
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_files.size(); }

    bool isEmpty() const { return m_files.isEmpty(); }
    Vector<String> paths() const;

    const Vector<Ref<File>>& files() const { return m_files; }
    const File& file(unsigned index) const { return m_files[index].get(); }

private:
    FileList() = default;
    FileList(Vector<Ref<File>>&& files)
        : m_files(WTFMove(files))
    {
    }

    // FileLists can only be changed by their owners.
    friend class DataTransfer;
    friend class FileInputType;
    void append(Ref<File>&& file) { m_files.append(WTFMove(file)); }
    void clear() { m_files.clear(); }

    Vector<Ref<File>> m_files;
};

} // namespace WebCore
