/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "OpenSSLHelper.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CurlSSLVerifier {
    WTF_MAKE_TZONE_ALLOCATED(CurlSSLVerifier);
    WTF_MAKE_NONCOPYABLE(CurlSSLVerifier);
public:
    CurlSSLVerifier(void* sslCtx);

    std::unique_ptr<WebCore::CertificateInfo> createCertificateInfo(std::optional<long>&&);

private:
    static int verifyCallback(int, X509_STORE_CTX*);
    void collectInfo(X509_STORE_CTX*);

    CertificateInfo::CertificateChain m_certificateChain;
};

} // namespace WebCore
