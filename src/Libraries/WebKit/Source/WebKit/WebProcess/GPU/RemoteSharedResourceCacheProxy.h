/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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

#if ENABLE(GPU_PROCESS)

#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebKit {

// Class for maintaining view of GPU process RemoteSharedResourceCache state in Web process.
// Thread-safe.
class RemoteSharedResourceCacheProxy : public ThreadSafeRefCounted<RemoteSharedResourceCacheProxy> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteSharedResourceCacheProxy);
public:
    static Ref<RemoteSharedResourceCacheProxy> create();
    virtual ~RemoteSharedResourceCacheProxy();

private:
    RemoteSharedResourceCacheProxy();
};

}

#endif
