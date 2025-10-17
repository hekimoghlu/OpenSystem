/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#include "InspectorExtensionTypes.h"

#if ENABLE(INSPECTOR_EXTENSIONS)

#include <wtf/text/WTFString.h>

namespace Inspector {

WTF::String extensionErrorToString(ExtensionError error)
{
    switch (error) {
    case ExtensionError::InternalError:
        return "InternalError"_s;
    case ExtensionError::InvalidRequest:
        return "InvalidRequest"_s;
    case ExtensionError::ContextDestroyed:
        return "ContextDestroyed"_s;
    case ExtensionError::RegistrationFailed:
        return "RegistrationFailed"_s;
    case ExtensionError::NotImplemented:
        return "NotImplemented"_s;
    }

    ASSERT_NOT_REACHED();
    return "InternalError"_s;
}

} // namespace WebKit

#endif // ENABLE(INSPECTOR_EXTENSIONS)
