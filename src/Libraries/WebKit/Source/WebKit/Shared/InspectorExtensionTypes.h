/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#if ENABLE(INSPECTOR_EXTENSIONS)

namespace API {
class SerializedScriptValue;
}

namespace WebCore {
struct ExceptionDetails;
}

namespace Inspector {
enum class ExtensionError : uint8_t;

using ExtensionTabID = WTF::String;
using ExtensionID = WTF::String;
using ExtensionVoidResult = Expected<void, ExtensionError>;
using ExtensionEvaluationResult = Expected<Expected<Ref<API::SerializedScriptValue>, WebCore::ExceptionDetails>, ExtensionError>;

enum class ExtensionAppearance : bool {
    Light,
    Dark
};

enum class ExtensionError : uint8_t {
    ContextDestroyed,
    InternalError,
    InvalidRequest,
    RegistrationFailed,
    NotImplemented,
};

WTF::String extensionErrorToString(ExtensionError);

} // namespace Inspector

#endif // ENABLE(INSPECTOR_EXTENSIONS)
