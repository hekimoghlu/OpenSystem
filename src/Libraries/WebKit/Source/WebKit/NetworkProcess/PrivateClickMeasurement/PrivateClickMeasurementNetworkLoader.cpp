/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#include "PrivateClickMeasurementNetworkLoader.h"

#include <WebCore/NotImplemented.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::PCM {

#if !PLATFORM(COCOA)

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkLoader);

void NetworkLoader::start(URL&&, RefPtr<JSON::Object>&&, WebCore::PrivateClickMeasurement::PcmDataCarried, Callback&& completionHandler)
{
    notImplemented();
    completionHandler({ }, { });
}

void NetworkLoader::allowTLSCertificateChainForLocalPCMTesting(const WebCore::CertificateInfo&)
{
    notImplemented();
}

#endif

} // namespace WebKit::PCM
