/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Icon;

enum class MediaCaptureType : uint8_t {
    MediaCaptureTypeNone,
    MediaCaptureTypeUser,
    MediaCaptureTypeEnvironment
};

struct FileChooserFileInfo {
    FileChooserFileInfo isolatedCopy() const & { return { path.isolatedCopy(), replacementPath.isolatedCopy(), displayName.isolatedCopy() }; }
    FileChooserFileInfo isolatedCopy() && { return { WTFMove(path).isolatedCopy(), WTFMove(replacementPath).isolatedCopy(), WTFMove(displayName).isolatedCopy() }; }

    String path;
    String replacementPath;
    String displayName;
};

struct FileChooserSettings {
    bool allowsDirectories { false };
    bool allowsMultipleFiles { false };
    Vector<String> acceptMIMETypes;
    Vector<String> acceptFileExtensions;
    Vector<String> selectedFiles;
#if ENABLE(MEDIA_CAPTURE)
    MediaCaptureType mediaCaptureType { MediaCaptureType::MediaCaptureTypeNone };
#endif
};

class FileChooserClient {
public:
    virtual ~FileChooserClient() = default;

    virtual void filesChosen(const Vector<FileChooserFileInfo>&, const String& displayString = { }, Icon* = nullptr) = 0;
    virtual void fileChoosingCancelled() = 0;
};

class FileChooser : public RefCounted<FileChooser> {
public:
    static Ref<FileChooser> create(FileChooserClient&, const FileChooserSettings&);
    WEBCORE_EXPORT ~FileChooser();

    void invalidate();

    WEBCORE_EXPORT void chooseFile(const String& path);
    WEBCORE_EXPORT void chooseFiles(const Vector<String>& paths, const Vector<String>& replacementPaths = { });
    WEBCORE_EXPORT void cancelFileChoosing();

#if PLATFORM(IOS_FAMILY)
    // FIXME: This function is almost identical to FileChooser::chooseFiles(). We should merge this in and remove this one.
    WEBCORE_EXPORT void chooseMediaFiles(const Vector<String>& paths, const String& displayString, Icon*);
#endif

    // FIXME: We should probably just pass file paths that could be virtual paths with proper display names rather than passing structs.
    void chooseFiles(const Vector<FileChooserFileInfo>& files);

    const FileChooserSettings& settings() const { return m_settings; }

private:
    FileChooser(FileChooserClient&, const FileChooserSettings&);

    FileChooserClient* m_client { nullptr };
    FileChooserSettings m_settings;
};

} // namespace WebCore
