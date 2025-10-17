/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "Clipboard.h"
#include <mutex>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

static Clipboard& clipboard()
{
    static std::once_flag onceFlag;
    static LazyNeverDestroyed<Clipboard> object;

    std::call_once(onceFlag, [] {
        object.construct(Clipboard::Type::Clipboard);
    });

    return object;
}

static Clipboard& primary()
{
    static std::once_flag onceFlag;
    static LazyNeverDestroyed<Clipboard> object;

    std::call_once(onceFlag, [] {
        object.construct(Clipboard::Type::Primary);
    });

    return object;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(Clipboard);

Clipboard& Clipboard::get(const String& name)
{
    if (name == "CLIPBOARD"_s)
        return clipboard();

    if (name == "PRIMARY"_s)
        return primary();

    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebKit
