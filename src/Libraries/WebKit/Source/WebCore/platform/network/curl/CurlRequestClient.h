/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

#include <wtf/AbstractRefCounted.h>
#include <wtf/Ref.h>

namespace WebCore {

class CertificateInfo;
class CurlRequest;
class CurlResponse;
class NetworkLoadMetrics;
class ResourceError;
class SharedBuffer;

class CurlRequestClient : public AbstractRefCounted {
public:
    virtual void curlDidSendData(CurlRequest&, unsigned long long bytesSent, unsigned long long totalBytesToBeSent) = 0;
    virtual void curlDidReceiveResponse(CurlRequest&, CurlResponse&&) = 0;
    virtual void curlDidReceiveData(CurlRequest&, Ref<SharedBuffer>&&) = 0;
    virtual void curlDidComplete(CurlRequest&, NetworkLoadMetrics&&) = 0;
    virtual void curlDidFailWithError(CurlRequest&, ResourceError&&, CertificateInfo&&) = 0;
};

} // namespace WebCore
