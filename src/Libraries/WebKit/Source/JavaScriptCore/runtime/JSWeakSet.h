/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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

#include "JSObject.h"
#include "WeakMapImpl.h"

namespace JSC {

class JSWeakSet final : public WeakMapImpl<WeakMapBucket<WeakMapBucketDataKey>> {
public:
    using Base = WeakMapImpl<WeakMapBucket<WeakMapBucketDataKey>>;

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSWeakSet* create(VM& vm, Structure* structure)
    {
        JSWeakSet* instance = new (NotNull, allocateCell<JSWeakSet>(vm)) JSWeakSet(vm, structure);
        instance->finishCreation(vm);
        return instance;
    }

private:
    JSWeakSet(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }
};

static_assert(std::is_final<JSWeakSet>::value, "Required for JSType based casting");

} // namespace JSC
