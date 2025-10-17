/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "RemoteLegacyCDMIdentifier.h"
#include <WebCore/LegacyCDMPrivate.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class Settings;
}

namespace WebKit {

class RemoteLegacyCDMFactory;
class RemoteLegacyCDMSession;


class RemoteLegacyCDM final
    : public WebCore::CDMPrivateInterface
    , public CanMakeWeakPtr<RemoteLegacyCDM> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLegacyCDM);
public:
    RemoteLegacyCDM(RemoteLegacyCDMFactory&, RemoteLegacyCDMIdentifier);
    virtual ~RemoteLegacyCDM();

    bool supportsMIMEType(const String&) const final;
    RefPtr<WebCore::LegacyCDMSession> createSession(WebCore::LegacyCDMSessionClient&) final;
    void setPlayerId(std::optional<WebCore::MediaPlayerIdentifier>);

    void ref() const final;
    void deref() const final;

private:
    Ref<RemoteLegacyCDMFactory> protectedFactory() const;

    WeakRef<RemoteLegacyCDMFactory> m_factory;
    RemoteLegacyCDMIdentifier m_identifier;
};

}

#endif
