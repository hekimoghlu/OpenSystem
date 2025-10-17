/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#ifndef _EAP8021X_EAPOLCLIENTCONFIGURATIONINTERNAL_H
#define _EAP8021X_EAPOLCLIENTCONFIGURATIONINTERNAL_H

#include <EAP8021X/EAPOLClientConfiguration.h>
#include <CoreFoundation/CFRuntime.h>
#include <SystemConfiguration/SCPreferences.h>
#include "symbol_scope.h"

#if TARGET_OS_IPHONE
typedef void AuthorizationExternalForm;
#endif

/*
 * EAPOLClientConfigurationInternal.h
 * - EAPOL client configuration internal data structures, functions
 */

/* 
 * Modification History
 *
 * December 8, 2009	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/* 
 * EAPOLClientConfiguration SCPreferences prefsID
 */
#define kEAPOLClientConfigurationPrefsID   CFSTR("com.apple.network.eapolclient.configuration.plist")

/**
 ** EAPOLClientConfiguration
 **/
struct __EAPOLClientConfiguration {
    CFRuntimeBase		cf_base;

    AuthorizationExternalForm *	auth_ext_p;
    SCPreferencesRef		eap_prefs;
    SCPreferencesRef		sc_prefs;
    CFMutableArrayRef		sc_changed_if;	/* of SCNetworkInterfaceRef */
    CFMutableDictionaryRef	profiles;	/* of EAPOLClientProfileRef */
    CFMutableDictionaryRef	ssids;		/* ssid -> profileID */
    CFMutableDictionaryRef	domains;	/* domain -> profileID */
    CFDictionaryRef		def_auth_props;	/* EAPClientProperties.h */
    Boolean			def_auth_props_changed;
    CFStringRef 		system_mode_profile_id; /* System Ethernet profileID */
    Boolean 			system_mode_profile_id_changed; /* System Ethernet profileID changed */
};

/**
 ** EAPOLClientProfile
 **/
struct __EAPOLClientProfile {
    CFRuntimeBase		cf_base;

    EAPOLClientConfigurationRef	cfg;
    CFStringRef			uuid;
    CFDictionaryRef		auth_props;
    CFStringRef			user_defined_name;
    struct {
	/* non HS 2.0 */
	CFDataRef		ssid;
	CFStringRef		security_type;
	
	/* HS 2.0 */
	CFStringRef		domain;
    } WLAN;
    CFMutableDictionaryRef	information;
};

/**
 ** EAPOLClientItemID
 **/
typedef enum {
    kEAPOLClientItemIDTypeNone = 0,
    kEAPOLClientItemIDTypeWLANSSID = 1,
    kEAPOLClientItemIDTypeProfileID = 2,
    kEAPOLClientItemIDTypeProfile = 3,
    kEAPOLClientItemIDTypeDefault = 4,
    kEAPOLClientItemIDTypeWLANDomain = 5
} EAPOLClientItemIDType;

struct __EAPOLClientItemID {
    CFRuntimeBase		cf_base;

    EAPOLClientItemIDType	type;
    union {
	CFDataRef		ssid;
	CFStringRef		profileID;
	EAPOLClientProfileRef	profile;
	CFStringRef		domain;
	const void *		ptr;
    } u;
};


/**
 ** EAPOLClientConfiguration functions
 **/
PRIVATE_EXTERN void
EAPOLClientConfigurationSetProfileForSSID(EAPOLClientConfigurationRef cfg,
					  CFDataRef ssid,
					  EAPOLClientProfileRef profile);
PRIVATE_EXTERN void
EAPOLClientConfigurationSetProfileForWLANDomain(EAPOLClientConfigurationRef cfg,
						CFStringRef domain,
						EAPOLClientProfileRef profile);
PRIVATE_EXTERN AuthorizationExternalForm *
EAPOLClientConfigurationGetAuthorizationExternalForm(EAPOLClientConfigurationRef cfg);

/**
 ** EAPOLClientProfile functions
 **/
PRIVATE_EXTERN EAPOLClientProfileRef
EAPOLClientProfileCreateWithDictAndProfileID(CFDictionaryRef dict,
					     CFStringRef profileID);
PRIVATE_EXTERN CFMutableDictionaryRef
EAPOLClientProfileCreateDictAndProfileID(EAPOLClientProfileRef profile,
					 CFStringRef * ret_profileID);
PRIVATE_EXTERN void
EAPOLClientProfileSetConfiguration(EAPOLClientProfileRef profile,
				   EAPOLClientConfigurationRef cfg);
PRIVATE_EXTERN EAPOLClientConfigurationRef
EAPOLClientProfileGetConfiguration(EAPOLClientProfileRef profile);

PRIVATE_EXTERN Boolean
accept_types_valid(CFArrayRef accept);

/**
 ** EAPOLClientItemID functions
 **/
PRIVATE_EXTERN AuthorizationExternalForm *
EAPOLClientItemIDGetAuthorizationExternalForm(EAPOLClientItemIDRef itemID);


#endif /* _EAP8021X_EAPOLCLIENTCONFIGURATIONINTERNAL_H */
