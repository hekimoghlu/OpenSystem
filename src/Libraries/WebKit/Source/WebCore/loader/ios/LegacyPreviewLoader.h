/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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

#include "PreviewConverterClient.h"
#include "PreviewConverterProvider.h"
#include "SharedBuffer.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class LegacyPreviewLoaderClient;
class ResourceLoader;
class ResourceResponse;

class LegacyPreviewLoader final : private PreviewConverterClient, private PreviewConverterProvider {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_MAKE_NONCOPYABLE(LegacyPreviewLoader);
public:
    LegacyPreviewLoader(ResourceLoader&, const ResourceResponse&);
    ~LegacyPreviewLoader();

    bool didReceiveResponse(const ResourceResponse&);
    bool didReceiveData(const SharedBuffer&);
    bool didFinishLoading();
    void didFail();

    WEBCORE_EXPORT static void setClientForTesting(RefPtr<LegacyPreviewLoaderClient>&&);

private:
    // PreviewConverterClient
    void previewConverterDidStartUpdating(PreviewConverter&) final { };
    void previewConverterDidFinishUpdating(PreviewConverter&) final { };
    void previewConverterDidFailUpdating(PreviewConverter&) final;
    void previewConverterDidStartConverting(PreviewConverter&) final;
    void previewConverterDidReceiveData(PreviewConverter&, const FragmentedSharedBuffer&) final;
    void previewConverterDidFinishConverting(PreviewConverter&) final;
    void previewConverterDidFailConverting(PreviewConverter&) final;

    // PreviewConverterProvider
    void provideMainResourceForPreviewConverter(PreviewConverter&, CompletionHandler<void(Ref<FragmentedSharedBuffer>&&)>&&) final;
    void providePasswordForPreviewConverter(PreviewConverter&, Function<void(const String&)>&&) final;

    RefPtr<PreviewConverter> protectedConverter() const;
    Ref<LegacyPreviewLoaderClient> protectedClient() const;

    RefPtr<PreviewConverter> m_converter;
    Ref<LegacyPreviewLoaderClient> m_client;
    SharedBufferBuilder m_originalData;
    WeakPtr<ResourceLoader> m_resourceLoader;
    bool m_finishedLoadingDataIntoConverter { false };
    bool m_hasProcessedResponse { false };
    bool m_needsToCallDidFinishLoading { false };
    bool m_shouldDecidePolicyBeforeLoading;
};

} // namespace WebCore
