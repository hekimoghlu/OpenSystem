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

#import <dispatch/dispatch.h>
#import <span>

namespace WTF {

#ifdef __OBJC__
inline std::span<const uint8_t> span(NSData *data)
{
    if (!data)
        return { };

    return unsafeMakeSpan(static_cast<const uint8_t*>(data.bytes), data.length);
}

inline RetainPtr<NSData> toNSData(std::span<const uint8_t> span)
{
    return adoptNS([[NSData alloc] initWithBytes:span.data() length:span.size()]);
}
#endif // #ifdef __OBJC__

template<typename> class Function;

#ifdef __cplusplus
extern "C" {
#endif
WTF_EXPORT_PRIVATE bool dispatch_data_apply_span(dispatch_data_t, const Function<bool(std::span<const uint8_t>)>& applier);
#ifdef __cplusplus
} // extern "C
#endif

} // namespace WTF

using WTF::dispatch_data_apply_span;

#ifdef __OBJC__
using WTF::span;
using WTF::toNSData;
#endif
