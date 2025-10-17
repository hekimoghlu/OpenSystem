/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#include "Base64Utilities.h"

#include <wtf/text/Base64.h>

namespace WebCore {

ExceptionOr<String> Base64Utilities::btoa(const String& stringToEncode)
{
    if (stringToEncode.isNull())
        return String();

    if (!stringToEncode.containsOnlyLatin1())
        return Exception { ExceptionCode::InvalidCharacterError };

    return base64EncodeToString(stringToEncode.latin1().span());
}

ExceptionOr<String> Base64Utilities::atob(const String& encodedString)
{
    if (encodedString.isNull())
        return String();

    auto decodedData = base64DecodeToString(encodedString, { Base64DecodeOption::ValidatePadding, Base64DecodeOption::IgnoreWhitespace });
    if (decodedData.isNull())
        return Exception { ExceptionCode::InvalidCharacterError };

    return decodedData;
}

}
