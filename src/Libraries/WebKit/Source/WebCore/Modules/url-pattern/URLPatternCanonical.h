/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#include "ExceptionOr.h"

namespace WebCore {

enum class BaseURLStringType : bool;
enum class EncodingCallbackType : uint8_t { Protocol, Username, Password, Host, IPv6Host, Port, Path, OpaquePath, Search, Hash };

bool isAbsolutePathname(StringView input, BaseURLStringType inputType);
ExceptionOr<String> canonicalizeProtocol(StringView, BaseURLStringType valueType);
String canonicalizeUsername(StringView value, BaseURLStringType valueType);
String canonicalizePassword(StringView value, BaseURLStringType valueType);
ExceptionOr<String> canonicalizeHostname(StringView value, BaseURLStringType valueType);
ExceptionOr<String> canonicalizeIPv6Hostname(StringView value, BaseURLStringType valueType);
ExceptionOr<String> canonicalizePort(StringView portValue, const std::optional<StringView> protocolValue, BaseURLStringType portValueType);
ExceptionOr<String> processPathname(StringView pathnameValue, const StringView protocolValue, BaseURLStringType pathnameValueType);
ExceptionOr<String> canonicalizePathname(StringView pathnameValue);
ExceptionOr<String> canonicalizeOpaquePathname(StringView value);
ExceptionOr<String> canonicalizeSearch(StringView value, BaseURLStringType valueType);
ExceptionOr<String> canonicalizeHash(StringView value, BaseURLStringType valueType);
ExceptionOr<String> callEncodingCallback(EncodingCallbackType, StringView input);
}
