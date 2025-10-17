/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#ifndef HEIMDAL_SYSTEM_CONFIGURATION_H
#define HEIMDAL_SYSTEM_CONFIGURATION_H 1

/**
 * SPI constants for OD/Heimdal communication for default realm and
 * location data using SystemConfiguration. This is a private
 * interface and can change any time.
 */

/**
 * Order array with list of default realm, first default realm is
 * listed first.
 */
#define HEIMDAL_SC_DEFAULT_REALM CFSTR("Kerberos-Default-Realms")

/**
 * Prefix for location of realm, append realm to key. Data is
 * Dictionary of types and then array of dict with host and port of
 * each servers within this type.
 */
#define HEIMDAL_SC_LOCATE_REALM_PREFIX CFSTR("Kerberos:")

/**
 * Locate type KDC
 */
#define HEIMDAL_SC_LOCATE_TYPE_KDC CFSTR("kdc")

/**
 * Locate type Kerberos change/set password
 */
#define HEIMDAL_SC_LOCATE_TYPE_KPASSWD CFSTR("kpasswd")

/**
 * Locate type Kerberos admin
 */
#define HEIMDAL_SC_LOCATE_TYPE_ADMIN CFSTR("kadmin")

/**
 *
 */
#define HEIMDAL_SC_LOCATE_PORT CFSTR("port")
#define HEIMDAL_SC_LOCATE_HOST CFSTR("host")


/**
 *
 */
#define HEIMDAL_SC_DOMAIN_REALM_MAPPING CFSTR("Kerberos-Domain-Realm-Mappings")




#endif /* HEIMDAL_SYSTEM_CONFIGURATION_H */
