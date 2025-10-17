/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include <WebCore/MediaPlayerEnums.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
struct MediaEngineSupportParameters;
}

namespace WebKit {

class RemoteMediaPlayerManager;

class RemoteMediaPlayerMIMETypeCache final : public CanMakeThreadSafeCheckedPtr<RemoteMediaPlayerMIMETypeCache> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaPlayerMIMETypeCache);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteMediaPlayerMIMETypeCache);
public:
    RemoteMediaPlayerMIMETypeCache(RemoteMediaPlayerManager&, WebCore::MediaPlayerEnums::MediaEngineIdentifier);
    ~RemoteMediaPlayerMIMETypeCache() = default;

    HashSet<String>& supportedTypes();
    WebCore::MediaPlayerEnums::SupportsType supportsTypeAndCodecs(const WebCore::MediaEngineSupportParameters&);
    void addSupportedTypes(const Vector<String>&);
    bool isEmpty() const;

private:
    Ref<RemoteMediaPlayerManager> protectedManager() const;

    ThreadSafeWeakPtr<RemoteMediaPlayerManager> m_manager; // Cannot be null.
    WebCore::MediaPlayerEnums::MediaEngineIdentifier m_engineIdentifier;

    using SupportedTypesAndCodecsKey = std::tuple<String, bool, bool, bool>;
    std::optional<HashMap<SupportedTypesAndCodecsKey, WebCore::MediaPlayerEnums::SupportsType>> m_supportsTypeAndCodecsCache;
    HashSet<String> m_supportedTypesCache;
    bool m_hasPopulatedSupportedTypesCacheFromGPUProcess { false };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
