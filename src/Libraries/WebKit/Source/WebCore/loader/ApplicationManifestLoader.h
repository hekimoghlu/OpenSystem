/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#if ENABLE(APPLICATION_MANIFEST)

#include "ApplicationManifest.h"
#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include "LoaderMalloc.h"
#include <wtf/CheckedRef.h>
#include <wtf/Noncopyable.h>
#include <wtf/URL.h>

namespace WebCore {

class CachedApplicationManifest;
class DocumentLoader;

class ApplicationManifestLoader final : private CachedRawResourceClient {
WTF_MAKE_NONCOPYABLE(ApplicationManifestLoader); WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    typedef Function<void (CachedResourceHandle<CachedApplicationManifest>)> CompletionHandlerType;

    ApplicationManifestLoader(DocumentLoader&, const URL&, bool);
    virtual ~ApplicationManifestLoader();

    bool startLoading();
    void stopLoading();

    std::optional<ApplicationManifest>& processManifest();

private:
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess);

    SingleThreadWeakRef<DocumentLoader> m_documentLoader;
    std::optional<ApplicationManifest> m_processedManifest;
    URL m_url;
    bool m_useCredentials;
    CachedResourceHandle<CachedApplicationManifest> m_resource;
};

} // namespace WebCore

#endif // ENABLE(APPLICATION_MANIFEST)
