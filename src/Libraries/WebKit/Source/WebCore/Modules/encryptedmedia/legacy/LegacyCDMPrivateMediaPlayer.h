/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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

#include "LegacyCDMPrivate.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class LegacyCDM;

class CDMPrivateMediaPlayer final : public CDMPrivateInterface {
    WTF_MAKE_TZONE_ALLOCATED(CDMPrivateMediaPlayer);
public:
    explicit CDMPrivateMediaPlayer(LegacyCDM& cdm)
        : m_cdm(cdm)
    { }

    static bool supportsKeySystem(const String&);
    static bool supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType);

    virtual ~CDMPrivateMediaPlayer() = default;

    bool supportsMIMEType(const String& mimeType) const override;
    RefPtr<LegacyCDMSession> createSession(LegacyCDMSessionClient&) override;

    LegacyCDM& cdm() const { return m_cdm; }

    void ref() const final;
    void deref() const final;

private:
    WeakRef<LegacyCDM> m_cdm;
};

} // namespace WebCore

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
