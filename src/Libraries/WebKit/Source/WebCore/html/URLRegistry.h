/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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

#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ScriptExecutionContext;
class SecurityOriginData;
class URLRegistry;

class URLRegistrable {
public:
    virtual ~URLRegistrable() = default;
    virtual URLRegistry& registry() const = 0;
    enum class RegistrableType : uint8_t { Blob, MediaSource };
    virtual RegistrableType registrableType() const = 0;
};

class URLRegistry {
    WTF_MAKE_TZONE_ALLOCATED(URLRegistry);
public:
    static void forEach(const Function<void(URLRegistry&)>&);

    URLRegistry();

    virtual ~URLRegistry();
    virtual void registerURL(const ScriptExecutionContext&, const URL&, URLRegistrable&) = 0;
    virtual void unregisterURL(const URL&, const SecurityOriginData& topOrigin) = 0;
    virtual void unregisterURLsForContext(const ScriptExecutionContext&) = 0;

    // This is an optional API
    virtual URLRegistrable* lookup(const String&) const { ASSERT_NOT_REACHED(); return 0; }
};

} // namespace WebCore
