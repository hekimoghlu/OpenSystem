/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#include "WebCoreBundleWin.h"

#include "WebCoreInstanceHandle.h"
#include <windows.h>
#include <wtf/FileSystem.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static String dllDirectory()
{
    WCHAR buffer[MAX_PATH];
    DWORD length = ::GetModuleFileNameW(WebCore::instanceHandle(), buffer, MAX_PATH);
    if (!length || (length == MAX_PATH && GetLastError() == ERROR_INSUFFICIENT_BUFFER))
        return emptyString();

    String path(buffer, length);
    return FileSystem::parentPath(path);
}

String webKitBundlePath()
{
    static NeverDestroyed<String> bundle = FileSystem::pathByAppendingComponent(dllDirectory(), "WebKit.resources"_s);
    return bundle;
}

String webKitBundlePath(StringView path)
{
    auto resource = FileSystem::pathByAppendingComponent(webKitBundlePath(), path);
    if (!FileSystem::fileExists(resource))
        return nullString();

    return resource;
}

String webKitBundlePath(StringView name, StringView type, StringView directory)
{
    auto fileName = makeString(name, '.', type);
    return webKitBundlePath(FileSystem::pathByAppendingComponent(directory, fileName));
}

String webKitBundlePath(const Vector<StringView>& components)
{
    return webKitBundlePath(FileSystem::pathByAppendingComponents(emptyString(), components));
}

} // namespace WebCore
