/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#ifndef CDMPrivateMediaSourceAVFObjC_h
#define CDMPrivateMediaSourceAVFObjC_h

#if ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)

#include "LegacyCDMPrivate.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class LegacyCDM;
class CDMSessionMediaSourceAVFObjC;

class CDMPrivateMediaSourceAVFObjC final : public CDMPrivateInterface, public CanMakeWeakPtr<CDMPrivateMediaSourceAVFObjC> {
    WTF_MAKE_TZONE_ALLOCATED(CDMPrivateMediaSourceAVFObjC);
public:
    explicit CDMPrivateMediaSourceAVFObjC(LegacyCDM& cdm)
        : m_cdm(cdm)
    { }
    virtual ~CDMPrivateMediaSourceAVFObjC();

    static bool supportsKeySystem(const String&);
    static bool supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType);

    bool supportsMIMEType(const String& mimeType) const override;
    RefPtr<LegacyCDMSession> createSession(LegacyCDMSessionClient&) override;

    LegacyCDM& cdm() const { return m_cdm.get(); }

    void invalidateSession(CDMSessionMediaSourceAVFObjC*);

    void ref() const final;
    void deref() const final;

private:
    struct KeySystemParameters {
        int version;
        Vector<int> protocols;
    };
    static std::optional<KeySystemParameters> parseKeySystem(const String& keySystem);
    
    WeakRef<LegacyCDM> m_cdm;
    Vector<CDMSessionMediaSourceAVFObjC*> m_sessions;
};

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA) && ENABLE(MEDIA_SOURCE)

#endif // CDMPrivateMediaSourceAVFObjC_h
