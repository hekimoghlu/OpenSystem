/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#import "SpanCocoa.h"

#import <wtf/BlockPtr.h>
#import <wtf/Function.h>
#import <wtf/StdLibExtras.h>

namespace WTF {

bool dispatch_data_apply_span(dispatch_data_t data, const Function<bool(std::span<const uint8_t>)>& applier)
{
    return dispatch_data_apply(data, makeBlockPtr([&applier](dispatch_data_t, size_t, const void* data, size_t size) {
        return applier(unsafeMakeSpan(static_cast<const uint8_t*>(data), size));
    }).get());
}

} // namespace WTF
