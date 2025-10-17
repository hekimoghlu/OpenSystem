/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "FileChooser.h"

namespace WebCore {

FileChooser::FileChooser(FileChooserClient& client, const FileChooserSettings& settings)
    : m_client(&client)
    , m_settings(settings)
{
}

Ref<FileChooser> FileChooser::create(FileChooserClient& client, const FileChooserSettings& settings)
{
    return adoptRef(*new FileChooser(client, settings));
}

FileChooser::~FileChooser() = default;

void FileChooser::invalidate()
{
    ASSERT(m_client);

    m_client = nullptr;
}

void FileChooser::chooseFile(const String& filename)
{
    chooseFiles({ filename });
}

void FileChooser::chooseFiles(const Vector<String>& filenames, const Vector<String>& replacementNames)
{
    if (!m_client)
        return;

    Vector<FileChooserFileInfo> files(filenames.size(), [&](size_t i) {
        return FileChooserFileInfo { filenames[i], i < replacementNames.size() ? replacementNames[i] : nullString(), { } };
    });
    m_client->filesChosen(WTFMove(files));
}

void FileChooser::cancelFileChoosing()
{
    if (!m_client)
        return;

    m_client->fileChoosingCancelled();
}

#if PLATFORM(IOS_FAMILY)

void FileChooser::chooseMediaFiles(const Vector<String>& filenames, const String& displayString, Icon* icon)
{
    if (!m_client)
        return;

    auto files = filenames.map([](auto& filename) {
        return FileChooserFileInfo { filename, { }, { } };
    });
    m_client->filesChosen(WTFMove(files), displayString, icon);
}

#endif

void FileChooser::chooseFiles(const Vector<FileChooserFileInfo>& files)
{
    if (m_client)
        m_client->filesChosen(files);
}

} // namespace WebCore
