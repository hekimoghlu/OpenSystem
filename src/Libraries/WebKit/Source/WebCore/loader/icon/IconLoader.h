/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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

#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include "LoaderMalloc.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/URL.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class CachedRawResource;
class DocumentLoader;
class LocalFrame;

class IconLoader final : private CachedRawResourceClient {
    WTF_MAKE_NONCOPYABLE(IconLoader); WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    IconLoader(DocumentLoader&, const URL&);
    virtual ~IconLoader();

    void startLoading();
    void stopLoading();

private:
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;

    SingleThreadWeakRef<DocumentLoader> m_documentLoader;
    URL m_url;
    CachedResourceHandle<CachedRawResource> m_resource;
};

} // namespace WebCore
