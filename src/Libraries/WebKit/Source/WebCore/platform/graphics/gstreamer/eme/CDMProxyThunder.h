/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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

#if ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER)

#include "CDMInstanceSession.h"
#include "CDMThunder.h"
#include "GStreamerEMEUtilities.h"
#include "MediaPlayerPrivate.h"
#include "SharedBuffer.h"
#include <wtf/Condition.h>
#include <wtf/VectorHash.h>

namespace WebCore {

// This is the thread-safe API that decode threads should use to make use of a platform CDM module.
class CDMProxyThunder final : public CDMProxy, public CanMakeWeakPtr<CDMProxyThunder, WeakPtrFactoryInitialization::Eager> {
public:
    CDMProxyThunder(const String& keySystem)
        : m_keySystem(keySystem) { }
    virtual ~CDMProxyThunder() = default;

    struct DecryptionContext {
        GstBuffer* keyIDBuffer;
        GstBuffer* ivBuffer;
        GstBuffer* dataBuffer;
        GstBuffer* subsamplesBuffer;
        size_t numSubsamples;
        WeakPtr<CDMProxyDecryptionClient> cdmProxyDecryptionClient;
    };

    bool decrypt(DecryptionContext&);
    const String& keySystem() { return m_keySystem; }

private:
    BoxPtr<OpenCDMSession> getDecryptionSession(DecryptionContext&) const;
    String m_keySystem;
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER)
