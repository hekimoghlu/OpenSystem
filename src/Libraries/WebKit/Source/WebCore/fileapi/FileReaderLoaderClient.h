/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#include "ExceptionCode.h"
#include "SharedBuffer.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
class FileReaderLoaderClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::FileReaderLoaderClient> : std::true_type { };
}

namespace WebCore {

class FileReaderLoaderClient : public CanMakeWeakPtr<FileReaderLoaderClient> {
public:
    virtual ~FileReaderLoaderClient() = default;

    virtual void didStartLoading() = 0;
    virtual void didReceiveData() = 0;
    virtual void didReceiveBinaryChunk(const SharedBuffer&) { }
    virtual void didFinishLoading() = 0;
    virtual void didFail(ExceptionCode errorCode) = 0;
};

} // namespace WebCore
