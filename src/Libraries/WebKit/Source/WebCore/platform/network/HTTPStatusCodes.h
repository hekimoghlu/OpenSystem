/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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

namespace WebCore {

// https://www.iana.org/assignments/http-status-codes/http-status-codes.xhtml

constexpr auto httpStatus103EarlyHints = 103;

constexpr auto httpStatus200OK = 200;
constexpr auto httpStatus204NoContent = 204;
constexpr auto httpStatus206PartialContent = 206;

constexpr auto httpStatus300MultipleChoices = 300;
constexpr auto httpStatus301MovedPermanently = 301;
constexpr auto httpStatus302Found = 302;
constexpr auto httpStatus303SeeOther = 303;
constexpr auto httpStatus304NotModified = 304;
constexpr auto httpStatus307TemporaryRedirect = 307;
constexpr auto httpStatus308PermanentRedirect = 308;

constexpr auto httpStatus400BadRequest = 400;
constexpr auto httpStatus401Unauthorized = 401;
constexpr auto httpStatus403Forbidden = 403;
constexpr auto httpStatus407ProxyAuthenticationRequired = 407;
constexpr auto httpStatus416RangeNotSatisfiable = 416;

} // namespace WebCore

using WebCore::httpStatus103EarlyHints;

using WebCore::httpStatus200OK;
using WebCore::httpStatus204NoContent;
using WebCore::httpStatus206PartialContent;

using WebCore::httpStatus300MultipleChoices;
using WebCore::httpStatus301MovedPermanently;
using WebCore::httpStatus302Found;
using WebCore::httpStatus303SeeOther;
using WebCore::httpStatus304NotModified;
using WebCore::httpStatus307TemporaryRedirect;
using WebCore::httpStatus308PermanentRedirect;

using WebCore::httpStatus400BadRequest;
using WebCore::httpStatus401Unauthorized;
using WebCore::httpStatus403Forbidden;
using WebCore::httpStatus407ProxyAuthenticationRequired;
using WebCore::httpStatus416RangeNotSatisfiable;
