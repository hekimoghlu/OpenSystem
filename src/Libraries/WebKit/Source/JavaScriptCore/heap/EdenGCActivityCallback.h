/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#include "GCActivityCallback.h"

namespace JSC {

class EdenGCActivityCallback : public GCActivityCallback {
public:
    static RefPtr<EdenGCActivityCallback> tryCreate(JSC::Heap& heap, Synchronousness synchronousness = Synchronousness::Async)
    {
        return s_shouldCreateGCTimer ? adoptRef(new EdenGCActivityCallback(heap, synchronousness)) : nullptr;
    }

    JS_EXPORT_PRIVATE void doCollection(VM&) override;

    JS_EXPORT_PRIVATE EdenGCActivityCallback(Heap&, Synchronousness);
    JS_EXPORT_PRIVATE ~EdenGCActivityCallback();

private:
    JS_EXPORT_PRIVATE Seconds lastGCLength(Heap&) final;
    JS_EXPORT_PRIVATE double gcTimeSlice(size_t bytes) final;
    JS_EXPORT_PRIVATE double deathRate(Heap&) final;
};

} // namespace JSC
