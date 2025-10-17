/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include "config.h"
#include "APIObject.h"

#include "WebKit2Initialize.h"

#include <wtf/ThreadSpecific.h>

#define DELEGATE_REF_COUNTING_TO_COCOA PLATFORM(COCOA)

namespace API {

#if DELEGATE_REF_COUNTING_TO_COCOA

HashMap<Object*, CFTypeRef>& Object::apiObjectsUnderConstruction()
{
    static LazyNeverDestroyed<ThreadSpecific<HashMap<Object*, CFTypeRef>>> s_apiObjectsUnderConstruction;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        s_apiObjectsUnderConstruction.construct();
    });
    return *s_apiObjectsUnderConstruction.get();
}

#endif

Object::Object()
#if DELEGATE_REF_COUNTING_TO_COCOA
    : m_wrapper(apiObjectsUnderConstruction().take(this))
#endif
{
#if DELEGATE_REF_COUNTING_TO_COCOA
    ASSERT(m_wrapper);
#endif
    WebKit::InitializeWebKit2();
}

} // namespace API
