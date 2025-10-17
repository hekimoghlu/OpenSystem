/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#include "CachedResource.h"

#if ENABLE(VIDEO)

namespace WebCore {

class CachedTextTrack final : public CachedResource {
public:
    CachedTextTrack(CachedResourceRequest&&, PAL::SessionID, const CookieJar*);

private:
    bool mayTryReplaceEncodedData() const override { return true; }
    void updateBuffer(const FragmentedSharedBuffer&) override;
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) override;

    void doUpdateBuffer(const FragmentedSharedBuffer*);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedTextTrack, CachedResource::Type::TextTrackResource)

#endif // ENABLE(VIDEO)
