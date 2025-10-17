/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#ifndef WKPluginInformation_h
#define WKPluginInformation_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Plug-in module information keys */

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPluginInformationBundleIdentifierKey(void);

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPluginInformationBundleVersionKey(void);

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPluginInformationBundleShortVersionKey(void);

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPluginInformationPathKey(void);

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPluginInformationDisplayNameKey(void);

/* Value type: WKUInt64Ref */
WK_EXPORT WKStringRef WKPluginInformationDefaultLoadPolicyKey(void);

/* Value type: WKBooleanRef */
WK_EXPORT WKStringRef WKPluginInformationUpdatePastLastBlockedVersionIsKnownAvailableKey(void);

/* Value type: WKBooleanRef */
WK_EXPORT WKStringRef WKPluginInformationHasSandboxProfileKey(void);


/* Plug-in load specific information keys */

/* Value type: WKURLRef */
WK_EXPORT WKStringRef WKPluginInformationFrameURLKey(void);

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPluginInformationMIMETypeKey(void);

/* Value type: WKURLRef */
WK_EXPORT WKStringRef WKPluginInformationPageURLKey(void);

/* Value type: WKURLRef */
WK_EXPORT WKStringRef WKPluginInformationPluginspageAttributeURLKey(void);

/* Value type: WKURLRef */
WK_EXPORT WKStringRef WKPluginInformationPluginURLKey(void);

/* Value type: WKBooleanRef */
WK_EXPORT WKStringRef WKPlugInInformationReplacementObscuredKey(void);

#ifdef __cplusplus
}
#endif

#endif /* WKPluginInformation_h */
