/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#ifndef _EAP8021X_EAPOLCONTROLTYPES_H
#define _EAP8021X_EAPOLCONTROLTYPES_H

#include <stdint.h>

#include <CoreFoundation/CFString.h>

enum {
    kEAPOLControlStateIdle,
    kEAPOLControlStateStarting,
    kEAPOLControlStateRunning,
    kEAPOLControlStateStopping,
};

typedef uint32_t EAPOLControlState;

/*
 * Property: kEAPOLControlEAPClientConfiguration
 * Purpose:
 *   The name of the sub-dictionary that contains the
 *   EAP client configuration parameters (keys defined in
 *   <EAP8021X/EAPClientProperties.h>).
 */
#define kEAPOLControlEAPClientConfiguration	CFSTR("EAPClientConfiguration")

/*
 * Property: kEAPOLControlUniqueIdentifier
 * Purpose:
 *   Mark the configuration with a unique string so that the
 *   UI can match it to a stored preference.
 *
 *   This property is also published as part of the status dictionary.
 */
#define kEAPOLControlUniqueIdentifier	CFSTR("UniqueIdentifier") /* CFString */

/*
 * Property: kEAPOLControlLogLevel
 * Purpose:
 *   Set the log level.  If the property is not present,
 *   logging is disabled.
 * Note:
 *   Deprecated.
 */
#define kEAPOLControlLogLevel		CFSTR("LogLevel") /* CFNumber */


/*
 * Property: kEAPOLControlEnableUserInterface
 * Purpose:
 *   Controls whether a user interface (UI) will be presented by the
 *   EAPOL client when information is required e.g. a missing name or password.
 *
 *   The default value is true.  When this is set to false, the EAPOL client
 *   will not present UI.
 */
#define kEAPOLControlEnableUserInterface \
    CFSTR("EnableUserInterface") /* CFBoolean */	

/*
 * properties that appear in the status dictionary
 */
#define kEAPOLControlIdentityAttributes	CFSTR("IdentityAttributes") /* CFArray(CFString) */
#define kEAPOLControlEAPType		CFSTR("EAPType")	/* CFNumber (EAPType) */
#define kEAPOLControlEAPTypeName	CFSTR("EAPTypeName")	/* CFString */
#define kEAPOLControlSupplicantState	CFSTR("SupplicantState") /* CFNumber (SupplicantState) */
#define kEAPOLControlClientStatus	CFSTR("ClientStatus")	/* CFNumber (EAPClientStatus) */
#define kEAPOLControlDomainSpecificError	CFSTR("DomainSpecificError") /* CFNumber (EAPClientDomainSpecificError) */
#define kEAPOLControlTimestamp		CFSTR("Timestamp")	/* CFDate */
#define kEAPOLControlLastStatusChangeTimestamp	CFSTR("LastStatusTimestamp")	/* CFDate */
#define kEAPOLControlRequiredProperties	CFSTR("RequiredProperties") /* CFArray[CFString] */
#define kEAPOLControlAdditionalProperties	CFSTR("AdditionalProperties") /* CFDictionary */
#define kEAPOLControlAuthenticatorMACAddress	CFSTR("AuthenticatorMACAddress") /* CFData */
#define kEAPOLControlManagerName	CFSTR("ManagerName")
#define kEAPOLControlUID		CFSTR("UID")


/*
 * Property: kEAPOLControlMode
 * Purpose:
 * - indicates which mode the EAPOL client is running in
 * - deprecates kEAPOLControlSystemMode (see below)
 */
enum {
    kEAPOLControlModeNone		= 0,
    kEAPOLControlModeUser		= 1,
    kEAPOLControlModeLoginWindow 	= 2,
    kEAPOLControlModeSystem		= 3
};
typedef uint32_t	EAPOLControlMode;

#define kEAPOLControlMode		CFSTR("Mode") /* CFNumber (EAPOLControlMode) */

/*
 * kEAPOLControlConfigurationGeneration
 * - the generation of the configuration that the client is using
 * - this value will be incremented when the client's configuration is
 *   changed i.e. as the result of calling EAPOLControlUpdate() or 
 *   EAPOLControlProvideUserInput()
 */
#define kEAPOLControlConfigurationGeneration \
    CFSTR("ConfigurationGeneration") /* CFNumber */

/*
 * Deprecated:
 */
#define kEAPOLControlSystemMode		CFSTR("SystemMode") /* CFBoolean */
#endif /* _EAP8021X_EAPOLCONTROLTYPES_H */
