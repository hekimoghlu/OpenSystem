/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#include "CurlSSLHandle.h"

#if USE(CURL)

#if NEED_OPENSSL_THREAD_SUPPORT && OS(WINDOWS)
#include <wtf/Threading.h>
#endif

namespace WebCore {

CurlSSLHandle::CurlSSLHandle()
{
#if NEED_OPENSSL_THREAD_SUPPORT
    ThreadSupport::setup();
#endif

    platformInitialize();
}

void CurlSSLHandle::setCACertPath(String&& caCertPath)
{
    RELEASE_ASSERT(!caCertPath.isEmpty());
    m_caCertInfo = WTFMove(caCertPath);
}

void CurlSSLHandle::setCACertData(CertificateInfo::Certificate&& caCertData)
{
    RELEASE_ASSERT(!caCertData.isEmpty());
    m_caCertInfo = WTFMove(caCertData);
}

void CurlSSLHandle::clearCACertInfo()
{
    m_caCertInfo = std::monostate { };
}

#if NEED_OPENSSL_THREAD_SUPPORT

void CurlSSLHandle::ThreadSupport::setup()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        singleton();
    });
}

CurlSSLHandle::ThreadSupport::ThreadSupport()
{
    CRYPTO_set_locking_callback(lockingCallback);
#if OS(WINDOWS)
    CRYPTO_THREADID_set_callback(threadIdCallback);
#endif
}

void CurlSSLHandle::ThreadSupport::lockingCallback(int mode, int type, const char*, int)
{
    RELEASE_ASSERT(type >= 0 && type < CRYPTO_NUM_LOCKS);
    auto& locker = ThreadSupport::singleton();

    if (mode & CRYPTO_LOCK)
        locker.lock(type);
    else
        locker.unlock(type);
}

#if OS(WINDOWS)

void CurlSSLHandle::ThreadSupport::threadIdCallback(CRYPTO_THREADID* threadId)
{
    CRYPTO_THREADID_set_numeric(threadId, static_cast<unsigned long>(Thread::currentID()));
}

#endif

#endif

}

#endif
