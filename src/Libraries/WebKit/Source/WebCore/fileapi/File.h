/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

#include "Blob.h"
#include <wtf/FileSystem.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class File final : public Blob {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(File, WEBCORE_EXPORT);
public:
    struct PropertyBag : BlobPropertyBag {
        std::optional<int64_t> lastModified;
    };

    // Create a file with an optional name exposed to the author (via File.name and associated DOM properties) that differs from the one provided in the path.
    WEBCORE_EXPORT static Ref<File> create(ScriptExecutionContext*, const String& path, const String& replacementPath = { }, const String& nameOverride = { }, const std::optional<FileSystem::PlatformFileID>& fileID = { });

    // Create a File using the 'new File' constructor.
    static Ref<File> create(ScriptExecutionContext& context, Vector<BlobPartVariant>&& blobPartVariants, const String& filename, const PropertyBag& propertyBag)
    {
        auto file = adoptRef(*new File(context, WTFMove(blobPartVariants), filename, propertyBag));
        file->suspendIfNeeded();
        return file;
    }

    static Ref<File> deserialize(ScriptExecutionContext* context, const String& path, const URL& srcURL, const String& type, const String& name, const std::optional<int64_t>& lastModified = std::nullopt)
    {
        auto file = adoptRef(*new File(deserializationContructor, context, path, srcURL, type, name, lastModified));
        file->suspendIfNeeded();
        return file;
    }

    static Ref<File> create(ScriptExecutionContext* context, const Blob& blob, const String& name)
    {
        auto file = adoptRef(*new File(context, blob, name));
        file->suspendIfNeeded();
        return file;
    }

    static Ref<File> create(ScriptExecutionContext* context, const File& existingFile, const String& name)
    {
        auto file = adoptRef(*new File(context, existingFile, name));
        file->suspendIfNeeded();
        return file;
    }

    static Ref<File> createWithRelativePath(ScriptExecutionContext*, const String& path, const String& relativePath);

    bool isFile() const override { return true; }

    const String& path() const { return m_path; }
    const String& relativePath() const { return m_relativePath; }
    void setRelativePath(const String& relativePath) { m_relativePath = relativePath; }
    const String& name() const { return m_name; }
    WEBCORE_EXPORT int64_t lastModified() const; // Number of milliseconds since Epoch.
    const std::optional<int64_t>& lastModifiedOverride() const { return m_lastModifiedDateOverride; } // Number of milliseconds since Epoch.
    const std::optional<FileSystem::PlatformFileID> fileID() const { return m_fileID; }

    WEBCORE_EXPORT static String contentTypeForFile(const String& path);

#if ENABLE(FILE_REPLACEMENT)
    static bool shouldReplaceFile(const String& path);
#endif

    bool isDirectory() const;

private:
    WEBCORE_EXPORT explicit File(ScriptExecutionContext*, const String& path);
    File(ScriptExecutionContext*, URL&&, String&& type, String&& path, String&& name);
    File(ScriptExecutionContext&, Vector<BlobPartVariant>&& blobPartVariants, const String& filename, const PropertyBag&);
    File(ScriptExecutionContext*, URL&&, String&& type, String&& path, String&& name, const std::optional<FileSystem::PlatformFileID>&);
    File(ScriptExecutionContext*, const Blob&, const String& name);
    File(ScriptExecutionContext*, const File&, const String& name);

    File(DeserializationContructor, ScriptExecutionContext*, const String& path, const URL& srcURL, const String& type, const String& name, const std::optional<int64_t>& lastModified);

    static void computeNameAndContentType(const String& path, const String& nameOverride, String& effectiveName, String& effectiveContentType);
#if ENABLE(FILE_REPLACEMENT)
    static void computeNameAndContentTypeForReplacedFile(const String& path, const String& nameOverride, String& effectiveName, String& effectiveContentType);
#endif

    String m_path;
    String m_relativePath;
    String m_name;

    std::optional<int64_t> m_lastModifiedDateOverride;
    std::optional<FileSystem::PlatformFileID> m_fileID;
    mutable std::optional<bool> m_isDirectory;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::File)
    static bool isType(const WebCore::Blob& blob) { return blob.isFile(); }
SPECIALIZE_TYPE_TRAITS_END()
