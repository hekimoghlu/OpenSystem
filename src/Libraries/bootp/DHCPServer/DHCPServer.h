/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
/*
 * DHCPServer.h
 */
/* 
 * Modification History
 *
 * November 10, 2000 	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */
#ifndef _S_DHCPSERVER_H
#define _S_DHCPSERVER_H
#include <os/availability.h>
#include <CoreFoundation/CoreFoundation.h>

/**
 ** DHCP and NetBoot properties 
 **/
extern const CFStringRef	kDHCPSPropIdentifier
API_AVAILABLE(macos(10.3), ios(8.0));
extern const CFStringRef	kDHCPSPropName
API_AVAILABLE(macos(10.3), ios(8.0));

/**
 ** DHCP lease dictionary properties 
 **/
extern const CFStringRef	kDHCPSPropDHCPLease
API_AVAILABLE(macos(10.3), ios(8.0));

extern const CFStringRef	kDHCPSPropDHCPIPAddress
API_AVAILABLE(macos(10.3), ios(8.0));

extern const CFStringRef	kDHCPSPropDHCPHWAddress
API_AVAILABLE(macos(10.3), ios(8.0));

/** 
 ** NetBoot client dictionary properties 
 **/
extern const CFStringRef	kDHCPSPropNetBootArch
API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

extern const CFStringRef	kDHCPSPropNetBootSysid
API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

extern const CFStringRef	kDHCPSPropNetBootLastBootTime
API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

extern const CFStringRef	kDHCPSPropNetBootIPAddress
API_AVAILABLE(macos(10.3)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

/*
 * Key: kDHCPSPropNetBootImageIndex, kDHCPSPropNetBootImageKind, 
 *      kDHCPSPropNetBootImageIsInstall
 * Purpose:
 *   Gives the image index, kind, and install attributes of the image that
 *   the client is bound to.
 * Notes:
 *   kDHCPSPropNetBootImageIndex        CFNumber (1..65535)
 *   kDHCPSPropNetBootImageKind         CFNumber
 *                                       0 = Classic/Mac OS 9,
 *                                       1 = Mac OS X, 2 = Mac OS X Server,
 *                                       3 = Diagnostics
 *   kDHCPSPropNetBootImageIsInstall    CFBoolean
 */
extern const CFStringRef	kDHCPSPropNetBootImageIndex
API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

extern const CFStringRef	kDHCPSPropNetBootImageKind
API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

extern const CFStringRef	kDHCPSPropNetBootImageIsInstall
API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);


/*
 * Key: kDHCPSPropNetBootImageID
 * Purpose:
 *   Gives a string representation of the 32-bit boot image ID, consisting
 *   of the index, kind, and install attributes.  
 */
extern const CFStringRef	kDHCPSPropNetBootImageID
API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

/*
 * Function: DHCPSDHCPLeaseListCreate
 * Purpose:
 *   Returns an array of DHCP lease dictionaries, NULL if none
 *   available.
 * Note:
 *   Returned array must be released using CFRelease().
 */
extern CFArrayRef		DHCPSDHCPLeaseListCreate(void)
API_AVAILABLE(macos(10.3), ios(8.0));


/*
 * Const: DHCPSDHCPLeaseListNotificationKey
 * Purpose:
 *   Use with notify(3) to be notified when the DHCP lease list
 *   adds or removes a client binding.
 */
extern const char * 		DHCPSDHCPLeaseListNotificationKey
API_AVAILABLE(macos(10.10), ios(8.0));


/*
 * Function: DHCPSCopyDisabledInterfaces
 * Purpose:
 *   Retrieve the list of interfaces on which DHCP has been disabled.
 *   The DHCP server can be configured to detect the presence of another
 *   DHCP server on a link and disable DHCP on that link when another DHCP
 *   server is detected.
 *
 * Returns:
 *   NULL if DHCP hasn't been disabled on any interface,
 *   non-NULL array of string interface names otherwise.
 */
extern CFArrayRef		DHCPSCopyDisabledInterfaces(void)
API_AVAILABLE(macos(10.10), ios(8.0));


/*
 * Const: DHCPSDisabledInterfacesNotificationKey
 * Purpose:
 *   Use with notify(3) to be notified when the list of disabled
 *   interfaces changes. Use DHCPSCopyDisabledInterfaces() to retrieve
 *   the current list.
 */
extern const char *		DHCPSDisabledInterfacesNotificationKey
API_AVAILABLE(macos(10.10), ios(8.0));

/*
 * Function: DHCPSNetBootClientListCreate
 * Purpose:
 *   Returns an array of NetBoot client dictionaries, NULL if none
 *   available.
 * Note:
 *   Returned array must be released using CFRelease().
 */
extern CFArrayRef		DHCPSNetBootClientListCreate(void)
	API_AVAILABLE(macos(10.4))
	API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

#endif /* _S_DHCPSERVER_H */
