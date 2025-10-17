/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#include "CDMProxyThunder.h"

#if ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER)

#include "CDMThunder.h"
#include "Logging.h"
#include <open_cdm_adapter.h>
#include <wtf/ByteOrder.h>

GST_DEBUG_CATEGORY_EXTERN(webkitMediaThunderDecryptDebugCategory);
#define GST_CAT_DEFAULT webkitMediaThunderDecryptDebugCategory

namespace WebCore {

// NOTE: YouTube 2019 EME conformance tests expect this to be >=5s.
const Seconds s_licenseKeyResponseTimeout = Seconds(6);

BoxPtr<OpenCDMSession> CDMProxyThunder::getDecryptionSession(DecryptionContext& in) const
{
    GstMappedBuffer mappedKeyID(in.keyIDBuffer, GST_MAP_READ);
    if (!mappedKeyID) {
        GST_ERROR("Failed to map key ID buffer");
        return nullptr;
    }

    auto keyID = mappedKeyID.createVector();

    auto keyHandle = getOrWaitForKeyHandle(keyID, WTFMove(in.cdmProxyDecryptionClient));
    if (!keyHandle.has_value() || !keyHandle.value()->isStatusCurrentlyValid())
        return nullptr;

    KeyHandleValueVariant keyData = keyHandle.value()->value();
    ASSERT(std::holds_alternative<BoxPtr<OpenCDMSession>>(keyData));

    BoxPtr<OpenCDMSession> keyValue = std::get<BoxPtr<OpenCDMSession>>(keyData);

    if (!keyValue) {
        keyValue = adoptInBoxPtr(opencdm_get_system_session(&static_cast<const CDMInstanceThunder*>(instance())->thunderSystem(), keyID.data(),
            keyID.size(), s_licenseKeyResponseTimeout.millisecondsAs<uint32_t>()));
        ASSERT(keyValue);
        // takeValueIfDifferent takes and r-value ref of
        // KeyHandleValueVariant. We want to copy the BoxPtr when
        // passing it down, cause we return it from this method. If we
        // just don't move the BoxPtr, the const BoxPtr& constructor
        // will be used. Anyway, letting that subtlety to the compiler
        // could be misleading so we explicitly invoke the const
        // BoxPtr& constructor here.
        keyHandle.value()->takeValueIfDifferent(BoxPtr<OpenCDMSession>(keyValue));
    }

    return keyValue;
}

bool CDMProxyThunder::decrypt(CDMProxyThunder::DecryptionContext& input)
{
    BoxPtr<OpenCDMSession> session = getDecryptionSession(input);
    if (!session) {
        GST_WARNING("there is no valid session to decrypt for the provided key ID (or the operation was aborted)");
        return false;
    }

    GST_TRACE("decrypting");
    // Decrypt cipher.
    OpenCDMError errorCode = opencdm_gstreamer_session_decrypt(session->get(), input.dataBuffer, input.subsamplesBuffer, input.numSubsamples,
        input.ivBuffer, input.keyIDBuffer, 0);
    if (errorCode) {
        GST_ERROR("decryption failed, error code %X", errorCode);
        return false;
    }

    return true;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER)
