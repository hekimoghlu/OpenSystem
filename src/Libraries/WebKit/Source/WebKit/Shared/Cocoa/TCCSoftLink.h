/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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

#import <pal/spi/cocoa/TCCSPI.h>

#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(WebKit, TCC)

SOFT_LINK_CONSTANT_FOR_HEADER(WebKit, TCC, kTCCServiceAccessibility, CFStringRef)
SOFT_LINK_CONSTANT_FOR_HEADER(WebKit, TCC, kTCCServiceCamera, CFStringRef)
SOFT_LINK_CONSTANT_FOR_HEADER(WebKit, TCC, kTCCServiceMicrophone, CFStringRef)
SOFT_LINK_CONSTANT_FOR_HEADER(WebKit, TCC, kTCCServicePhotos, CFStringRef)
SOFT_LINK_CONSTANT_FOR_HEADER(WebKit, TCC, kTCCServiceWebKitIntelligentTrackingPrevention, CFStringRef)

SOFT_LINK_FUNCTION_FOR_HEADER(WebKit, TCC, TCCAccessCheckAuditToken, Boolean, (CFStringRef service, audit_token_t auditToken, CFDictionaryRef options), (service, auditToken, options))
#define TCCAccessCheckAuditToken WebKit::softLink_TCC_TCCAccessCheckAuditToken
SOFT_LINK_FUNCTION_FOR_HEADER(WebKit, TCC, TCCAccessPreflight, TCCAccessPreflightResult, (CFStringRef service, CFDictionaryRef options), (service, options))
#define TCCAccessPreflight WebKit::softLink_TCC_TCCAccessPreflight
SOFT_LINK_FUNCTION_FOR_HEADER(WebKit, TCC, TCCAccessPreflightWithAuditToken, TCCAccessPreflightResult, (CFStringRef service, audit_token_t token, CFDictionaryRef options), (service, token, options))
#define TCCAccessPreflightWithAuditToken WebKit::softLink_TCC_TCCAccessPreflightWithAuditToken
#if HAVE(TCC_IOS_14_BIG_SUR_SPI)
SOFT_LINK_FUNCTION_FOR_HEADER(WebKit, TCC, tcc_identity_create, tcc_identity_t, (tcc_identity_type_t type, const char * identifier), (type, identifier));
#define tcc_identity_create WebKit::softLink_TCC_tcc_identity_create
#endif // HAVE(TCC_IOS_14_BIG_SUR_SPI)
