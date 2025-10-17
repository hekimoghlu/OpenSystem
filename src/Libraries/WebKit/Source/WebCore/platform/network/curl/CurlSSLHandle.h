/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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

#include "CertificateInfo.h"
#include <openssl/crypto.h>
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringHash.h>

// all version of LibreSSL and OpenSSL prior to 1.1.0 need thread support
#if defined(LIBRESSL_VERSION_NUMBER) || (defined(OPENSSL_VERSION_NUMBER) && OPENSSL_VERSION_NUMBER < 0x1010000fL)
#define NEED_OPENSSL_THREAD_SUPPORT 1
#else
#define NEED_OPENSSL_THREAD_SUPPORT 0
#endif

namespace WebCore {

class CurlSSLHandle {
    WTF_MAKE_NONCOPYABLE(CurlSSLHandle);
    friend NeverDestroyed<CurlSSLHandle>;

public:
    using CACertInfo = std::variant<std::monostate, String, CertificateInfo::Certificate>;

    CurlSSLHandle();

    const CString& cipherList() const { return m_cipherList; }
    const CString& signatureAlgorithmsList() const { return m_signatureAlgorithmsList; }
    const CString& ecCurves() const { return m_ecCurves; }

    void setCipherList(CString&& data) { m_cipherList = WTFMove(data); }
    void setSignatureAlgorithmsList(CString&& data) { m_signatureAlgorithmsList = WTFMove(data); }
    void setECCurves(CString&& data) { m_ecCurves = WTFMove(data); }

    bool shouldIgnoreSSLErrors() const { return m_ignoreSSLErrors; }
    WEBCORE_EXPORT void setIgnoreSSLErrors(bool flag) { m_ignoreSSLErrors = flag; }

    const CACertInfo& getCACertInfo() const { return m_caCertInfo; }
    WEBCORE_EXPORT void setCACertPath(String&&);
    WEBCORE_EXPORT void setCACertData(CertificateInfo::Certificate&&);
    WEBCORE_EXPORT void clearCACertInfo();

private:
#if NEED_OPENSSL_THREAD_SUPPORT
    class ThreadSupport {
        friend NeverDestroyed<CurlSSLHandle::ThreadSupport>;
        
    public:
        static void setup();

    private:
        static ThreadSupport& singleton()
        {
            static NeverDestroyed<CurlSSLHandle::ThreadSupport> shared;
            return shared;
        }

        ThreadSupport();
        void lock(int type) { m_locks[type].lock(); }
        void unlock(int type) { m_locks[type].unlock(); }

        Lock m_locks[CRYPTO_NUM_LOCKS];

        static void lockingCallback(int mode, int type, const char* file, int line);
#if OS(WINDOWS)
        static void threadIdCallback(CRYPTO_THREADID*);
#endif
    };
#endif

    void platformInitialize();

    CString m_cipherList;
    CString m_signatureAlgorithmsList;
    CString m_ecCurves;
    CACertInfo m_caCertInfo;

    bool m_ignoreSSLErrors { false };
};

} // namespace WebCore
