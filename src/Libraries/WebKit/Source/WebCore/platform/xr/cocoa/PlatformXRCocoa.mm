/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
#import "config.h"
#import "PlatformXRCocoa.h"
#import <wtf/TZoneMallocInlines.h>

#if ENABLE(WEBXR) && !USE(OPENXR)

#import "PlatformXR.h"
#import <wtf/NeverDestroyed.h>

using namespace WebCore;

namespace PlatformXR {

struct Instance::Impl {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(Impl);
public:
    Impl() = default;
    ~Impl() = default;
};

Instance& Instance::singleton()
{
    static LazyNeverDestroyed<Instance> s_instance;
    static std::once_flag s_onceFlag;
    std::call_once(s_onceFlag,
        [&] {
            s_instance.construct();
        });
    return s_instance.get();
}

Instance::Instance()
    : m_impl(makeUniqueRef<Impl>())
{
}

void Instance::enumerateImmersiveXRDevices(CompletionHandler<void(const DeviceList& devices)>&& callback)
{
    callback({ });
}

}

#endif
