/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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

#include <wtf/EnumTraits.h>

namespace WebCore { 

enum class FetchHeadersGuard : uint8_t {
    None,
    Immutable,
    Request,
    RequestNoCors,
    Response
};

} // namespace WebCore

namespace WTF {

template<> struct EnumTraitsForPersistence<WebCore::FetchHeadersGuard> {
    using values = EnumValues<
    WebCore::FetchHeadersGuard,
    WebCore::FetchHeadersGuard::None,
    WebCore::FetchHeadersGuard::Immutable,
    WebCore::FetchHeadersGuard::Request,
    WebCore::FetchHeadersGuard::RequestNoCors,
    WebCore::FetchHeadersGuard::Response
    >;
};

}
