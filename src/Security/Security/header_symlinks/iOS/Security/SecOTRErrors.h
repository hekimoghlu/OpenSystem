/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#ifndef messsageProtection_SecMessageProtectionErrors_h
#define messsageProtection_SecMessageProtectionErrors_h

#include <CoreFoundation/CoreFoundation.h>

static const CFIndex kSecOTRErrorFailedToEncrypt = -1;
static const CFIndex kSecOTRErrorFailedToDecrypt = -2;
static const CFIndex kSecOTRErrorFailedToVerify = -3;
static const CFIndex kSecOTRErrorFailedToSign = -4;
static const CFIndex kSecOTRErrorSignatureDidNotMatch = -5;
static const CFIndex kSecOTRErrorFailedSelfTest = -6;
static const CFIndex kSecOTRErrorParameterError = -7;
static const CFIndex kSecOTRErrorUnknownFormat = -8;
static const CFIndex kSecOTRErrorCreatePublicIdentity = -9;
static const CFIndex kSecOTRErrorCreatePublicBytes = -10;

// Errors 100-199 reserved for errors being genrated by workarounds/known issues failing
static const CFIndex kSecOTRErrorSignatureTooLarge = -100;
static const CFIndex kSecOTRErrorSignatureDidNotRecreate = -101;

#endif
