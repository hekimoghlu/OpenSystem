/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include <CoreFoundation/CFString.h>

#define kNetBootImageInfoArchitectures	CFSTR("Architectures")	/* Array[String] */
#define kNetBootImageInfoIndex		CFSTR("Index")		/* Number */
#define kNetBootImageInfoIsEnabled	CFSTR("IsEnabled") 	/* Boolean */
#define kNetBootImageInfoIsInstall	CFSTR("IsInstall")	/* Boolean */
#define kNetBootImageInfoName		CFSTR("Name")		/* String */
#define kNetBootImageInfoType		CFSTR("Type")		/* String */
#define kNetBootImageInfoBootFile	CFSTR("BootFile")	/* String */
#define kNetBootImageInfoIsDefault	CFSTR("IsDefault")	/* Boolean */
#define kNetBootImageInfoKind		CFSTR("Kind")		/* Number */
#define kNetBootImageInfoSupportsDiskless CFSTR("SupportsDiskless") /* Boolean */
#define kNetBootImageInfoEnabledSystemIdentifiers CFSTR("EnabledSystemIdentifiers") /* Array[String] */
#define kNetBootImageInfoFilterOnly 	CFSTR("FilterOnly")	/* Boolean */
#define kNetBootImageInfoEnabledMACAddresses CFSTR("EnabledMACAddresses") /* Array[String] */
#define kNetBootImageInfoDisabledMACAddresses CFSTR("DisabledMACAddresses") /* Array[String] */
#define kNetBootImageLoadBalanceServer 	CFSTR("LoadBalanceServer") /* String */


/* Type values */
#define kNetBootImageInfoTypeClassic	CFSTR("Classic")
#define kNetBootImageInfoTypeNFS	CFSTR("NFS")
#define kNetBootImageInfoTypeHTTP	CFSTR("HTTP")
#define kNetBootImageInfoTypeBootFileOnly CFSTR("BootFileOnly")

/* Classic specific keys */
#define kNetBootImageInfoPrivateImage	CFSTR("PrivateImage")	/* String */
#define kNetBootImageInfoSharedImage	CFSTR("SharedImage")	/* String */

/* NFS, HTTP specific keys */
#define kNetBootImageInfoRootPath	CFSTR("RootPath")	/* String */


