/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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

#include <WebCore/ExceptionOr.h>

NO_RETURN void raiseTypeErrorException();
NO_RETURN void raiseNotSupportedErrorException();

NO_RETURN void raiseDOMErrorException(WebCore::Exception&&);
template<typename T> T raiseOnDOMError(WebCore::ExceptionOr<T>&&);
void raiseOnDOMError(WebCore::ExceptionOr<void>&&);

inline void raiseOnDOMError(WebCore::ExceptionOr<void>&& possibleException)
{
    if (possibleException.hasException())
        raiseDOMErrorException(possibleException.releaseException());
}

template<typename T> inline T raiseOnDOMError(WebCore::ExceptionOr<T>&& exceptionOrReturnValue)
{
    if (exceptionOrReturnValue.hasException())
        raiseDOMErrorException(exceptionOrReturnValue.releaseException());
    return exceptionOrReturnValue.releaseReturnValue();
}
