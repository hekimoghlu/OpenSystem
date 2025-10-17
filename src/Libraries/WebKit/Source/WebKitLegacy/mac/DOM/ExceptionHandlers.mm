/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#import "ExceptionHandlers.h"

#import "DOMEventException.h"
#import "DOMException.h"
#import "DOMRangeException.h"
#import "DOMXPathException.h"
#import <WebCore/DOMException.h>

NSString * const DOMException = @"DOMException";
NSString * const DOMRangeException = @"DOMRangeException";
NSString * const DOMEventException = @"DOMEventException";
NSString * const DOMXPathException = @"DOMXPathException";

static NO_RETURN void raiseDOMErrorException(WebCore::ExceptionCode ec)
{
    ASSERT(static_cast<bool>(ec));

    auto description = WebCore::DOMException::description(ec);

    RetainPtr<NSString> reason;
    if (description.name)
        reason = adoptNS([[NSString alloc] initWithFormat:@"*** %s: %@ %d", description.name.characters(), DOMException, description.legacyCode]);
    else
        reason = adoptNS([[NSString alloc] initWithFormat:@"*** %@ %d", DOMException, description.legacyCode]);

    auto userInfo = @{ DOMException: @(description.legacyCode) };
    auto exception = [NSException exceptionWithName:DOMException reason:reason.get() userInfo:userInfo];
    [exception raise];

    RELEASE_ASSERT_NOT_REACHED();
}

void raiseTypeErrorException()
{
    raiseDOMErrorException(WebCore::ExceptionCode::TypeError);
}

void raiseNotSupportedErrorException()
{
    raiseDOMErrorException(WebCore::ExceptionCode::NotSupportedError);
}

void raiseDOMErrorException(WebCore::Exception&& exception)
{
    raiseDOMErrorException(exception.code());
}
