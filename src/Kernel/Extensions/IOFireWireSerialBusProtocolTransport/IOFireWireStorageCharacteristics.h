/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#ifndef _IOKIT_IO_FIREWIRE_STORAGE_DEVICE_CHARACTERISTICS_H_
#define _IOKIT_IO_FIREWIRE_STORAGE_DEVICE_CHARACTERISTICS_H_

//
//	Bridge Characteristics - Characteristics defined for FireWire bridges.
//

/*!
@defined kIOPropertyBridgeCharacteristicsKey
@discussion This key is used to define Bridge Characteristics for a particular
devices's bridge chipset. It has an associated dictionary which lists the
bridge characteristics.

Requirement: Optional

Example:
<pre>
@textblock
<dict>
	<key>Bridge Characteristics</key>
	<dict>
		<key>Bridge Vendor Name</key>
		<string>Oxford Semiconductor</string>
		<key>Bridge Model Name</key>
		<string>FW911</string>
		<key>Bridge Revision Level</key>
		<string>3.7</string>
	</dict>
</dict>
@/textblock
</pre>
*/

#define kIOPropertyBridgeCharacteristicsKey		"Bridge Characteristics"
#define kIOPropertyBridgeVendorNameKey			"Bridge Vendor Name"
#define kIOPropertyBridgeModelNameKey			"Bridge Model Name"
#define kIOPropertyBridgeRevisionLevelKey		"Bridge Revision Level"

#endif	/* _IOKIT_IO_FIREWIRE_STORAGE_DEVICE_CHARACTERISTICS_H_ */