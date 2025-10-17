/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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

#include "ActiveDOMObject.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DOMFileSystem;
class ErrorCallback;
class FileSystemEntryCallback;

class FileSystemEntry : public ScriptWrappable, public ActiveDOMObject, public RefCounted<FileSystemEntry> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FileSystemEntry);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    virtual ~FileSystemEntry();

    virtual bool isFile() const { return false; }
    virtual bool isDirectory() const { return false; }

    const String& name() const { return m_name; }
    const String& virtualPath() const { return m_virtualPath; }
    DOMFileSystem& filesystem() const;

    void getParent(ScriptExecutionContext&, RefPtr<FileSystemEntryCallback>&&, RefPtr<ErrorCallback>&&);

protected:
    FileSystemEntry(ScriptExecutionContext&, DOMFileSystem&, const String& virtualPath);
    Document* document() const;

private:
    Ref<DOMFileSystem> m_filesystem;
    String m_name;
    String m_virtualPath;
};

} // namespace WebCore
