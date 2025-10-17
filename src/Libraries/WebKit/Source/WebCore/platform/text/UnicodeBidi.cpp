/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include "UnicodeBidi.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, UnicodeBidi value)
{
    switch (value) {
    case UnicodeBidi::Normal: ts << "Normal"; break;
    case UnicodeBidi::Embed: ts << "Embed"; break;
    case UnicodeBidi::Override: ts << "Override"; break;
    case UnicodeBidi::Isolate: ts << "Isolate"; break;
    case UnicodeBidi::Plaintext: ts << "Plaintext"; break;
    case UnicodeBidi::IsolateOverride: ts << "IsolateOverride"; break;
    }
    return ts;
}

} // namespace WebCore
