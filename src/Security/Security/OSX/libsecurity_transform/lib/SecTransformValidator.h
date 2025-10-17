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
#ifndef _SEC_TRANSFORMVALIDATOR_H__
#define _SEC_TRANSFORMVALIDATOR_H__

#include <Security/SecCustomTransform.h>

CF_EXTERN_C_BEGIN

/*!
 @function			SecTransformCreateValidatorForCFtype
 
 @abstract			Create a validator that triggers an error when presented a CF type other then expected_type,
 or a NULL value if null_allowed is NO.
 
 @result				A SecTransformAttributeActionBlock suitable for passing to SecTransformSetAttributeAction
 with an actyion type of kSecTransformActionAttributeValidation.
 
 @discussion			If the validator is passed an incorrect CF type it will return a CFError including the
 type it was given, the value it was given, the type it expected, and if a NULL value is acceptable as well as
 what attribute the value was sent to.
 */
CF_EXPORT 
SecTransformAttributeActionBlock SecTransformCreateValidatorForCFtype(CFTypeID expected_type, Boolean null_allowed)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

CF_EXTERN_C_END

#endif
