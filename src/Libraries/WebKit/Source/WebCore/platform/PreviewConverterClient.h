/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
struct PreviewConverterClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PreviewConverterClient> : std::true_type { };
}

namespace WebCore {

class PreviewConverter;
class ResourceError;
class FragmentedSharedBuffer;

struct PreviewConverterClient : CanMakeWeakPtr<PreviewConverterClient> {
    virtual ~PreviewConverterClient() = default;

    virtual void previewConverterDidStartUpdating(PreviewConverter&) = 0;
    virtual void previewConverterDidFinishUpdating(PreviewConverter&) = 0;
    virtual void previewConverterDidFailUpdating(PreviewConverter&) = 0;
    virtual void previewConverterDidStartConverting(PreviewConverter&) = 0;
    virtual void previewConverterDidReceiveData(PreviewConverter&, const FragmentedSharedBuffer& newData) = 0;
    virtual void previewConverterDidFinishConverting(PreviewConverter&) = 0;
    virtual void previewConverterDidFailConverting(PreviewConverter&) = 0;
};

} // namespace WebCore
