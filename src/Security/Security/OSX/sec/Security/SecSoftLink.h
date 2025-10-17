/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
/*!
    @header SecSoftLink.h
    Contains declaration of framework and classes/constants softlinked into Security
*/

#ifndef _SECURITY_SECSOFTLINK_H_
#define _SECURITY_SECSOFTLINK_H_

#import <Foundation/Foundation.h>
#import <LocalAuthentication/LocalAuthentication_Private.h>
#import <CryptoTokenKit/CryptoTokenKit_Private.h>
#import <SoftLinking/SoftLinking.h>

SOFT_LINK_OPTIONAL_FRAMEWORK_FOR_HEADER(CryptoTokenKit);
SOFT_LINK_CLASS_FOR_HEADER(CryptoTokenKit, TKClientToken);
SOFT_LINK_CLASS_FOR_HEADER(CryptoTokenKit, TKClientTokenSession);
SOFT_LINK_OBJECT_CONSTANT_FOR_HEADER(CryptoTokenKit, TKErrorDomain);
SOFT_LINK_CONSTANT(CryptoTokenKit, TKClientTokenParameterForceSystemSession, NSString *)
SOFT_LINK_OPTIONAL_FRAMEWORK_FOR_HEADER(LocalAuthentication);
SOFT_LINK_CLASS_FOR_HEADER(LocalAuthentication, LAContext);
SOFT_LINK_OBJECT_CONSTANT_FOR_HEADER(LocalAuthentication, LAErrorDomain);

#endif /* !_SECURITY_SECSOFTLINK_H_ */
