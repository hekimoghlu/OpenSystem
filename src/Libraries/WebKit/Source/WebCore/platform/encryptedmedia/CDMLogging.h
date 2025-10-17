/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include <wtf/text/WTFString.h>

namespace WebCore {

struct CDMKeySystemConfiguration;
struct CDMMediaCapability;
struct CDMRestrictions;

enum class CDMEncryptionScheme : bool;
enum class CDMKeyStatus : uint8_t;
enum class CDMMessageType : uint8_t;
enum class CDMRequirement : uint8_t;
enum class CDMSessionType : uint8_t;

}

namespace WTF {

template<typename>
struct LogArgument;

template <>
struct LogArgument<WebCore::CDMKeySystemConfiguration> {
    static String toString(const WebCore::CDMKeySystemConfiguration&);
};

template <>
struct LogArgument<WebCore::CDMMediaCapability> {
    static String toString(const WebCore::CDMMediaCapability&);
};

template <>
struct LogArgument<WebCore::CDMRestrictions> {
    static String toString(const WebCore::CDMRestrictions&);
};

template <>
struct LogArgument<WebCore::CDMEncryptionScheme> {
    static String toString(const WebCore::CDMEncryptionScheme&);
};

template <>
struct LogArgument<WebCore::CDMKeyStatus> {
    static String toString(const WebCore::CDMKeyStatus&);
};

template <>
struct LogArgument<WebCore::CDMMessageType> {
    static String toString(const WebCore::CDMMessageType&);
};

template <>
struct LogArgument<WebCore::CDMRequirement> {
    static String toString(const WebCore::CDMRequirement&);
};

template <>
struct LogArgument<WebCore::CDMSessionType> {
    static String toString(const WebCore::CDMSessionType&);
};

}

#endif // ENABLE(ENCRYPTED_MEDIA)
