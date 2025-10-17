/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "WKProtectionSpaceCurl.h"

#include "APIArray.h"
#include "APIData.h"
#include "WKAPICast.h"
#include "WebCredential.h"
#include "WebProtectionSpace.h"
#include <WebCore/CertificateInfo.h>

WKCertificateInfoRef WKProtectionSpaceCopyCertificateInfo(WKProtectionSpaceRef protectionSpaceRef)
{
    return nullptr;
}

WKArrayRef WKProtectionSpaceCopyCertificateChain(WKProtectionSpaceRef protectionSpace)
{
    auto& certificateChain = WebKit::toImpl(protectionSpace)->protectionSpace().certificateInfo().certificateChain();
    auto certs = WTF::map(certificateChain, [](auto&& certificate) -> RefPtr<API::Object> {
        return API::Data::create(certificate.span());
    });
    return WebKit::toAPI(API::Array::create(WTFMove(certs)).leakRef());
}

int WKProtectionSpaceGetCertificateVerificationError(WKProtectionSpaceRef protectionSpace)
{
    auto& certificateInfo = WebKit::toImpl(protectionSpace)->protectionSpace().certificateInfo();
    return certificateInfo.verificationError();
}

WKStringRef WKProtectionSpaceCopyCertificateVerificationErrorDescription(WKProtectionSpaceRef protectionSpace)
{
    auto& certificateInfo = WebKit::toImpl(protectionSpace)->protectionSpace().certificateInfo();
    return WebKit::toCopiedAPI(certificateInfo.verificationErrorDescription());
}
