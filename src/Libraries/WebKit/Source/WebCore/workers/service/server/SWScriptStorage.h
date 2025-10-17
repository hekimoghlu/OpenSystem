/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#include <wtf/FileSystem.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ServiceWorkerRegistrationKey;
class ScriptBuffer;

class SWScriptStorage {
    WTF_MAKE_TZONE_ALLOCATED(SWScriptStorage);
public:
    explicit SWScriptStorage(const String& directory);

    ScriptBuffer store(const ServiceWorkerRegistrationKey&, const URL& scriptURL, const ScriptBuffer&);
    ScriptBuffer retrieve(const ServiceWorkerRegistrationKey&, const URL& scriptURL);
    void clear(const ServiceWorkerRegistrationKey&);

private:
    String registrationDirectory(const ServiceWorkerRegistrationKey&) const;
    String scriptPath(const ServiceWorkerRegistrationKey&, const URL& scriptURL) const;
    String saltPath() const;
    String sha2Hash(const String&) const;
    String sha2Hash(const URL&) const;

    String m_directory;
    FileSystem::Salt m_salt;
};

} // namespace WebCore
