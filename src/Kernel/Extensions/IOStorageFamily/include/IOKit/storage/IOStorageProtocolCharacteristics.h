/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#ifndef _IOKIT_IO_STORAGE_PROTOCOL_CHARACTERISTICS_H_
#define _IOKIT_IO_STORAGE_PROTOCOL_CHARACTERISTICS_H_

#include <TargetConditionals.h>

#if TARGET_OS_DRIVERKIT
#include <DriverKit/storage/IOStorageControllerCharacteristics.h>
#else
#include <IOKit/storage/IOStorageControllerCharacteristics.h>
#endif

/*
 *	Protocol Characteristics - Characteristics defined for protocols.
 */

/*!
@defined kIOPropertyProtocolCharacteristicsKey
@discussion This key is used to define Protocol Characteristics for a particular
protocol and it has an associated dictionary which lists the
protocol characteristics.

Requirement: Mandatory

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>ATAPI</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define	kIOPropertyProtocolCharacteristicsKey		"Protocol Characteristics"


/*!
@defined kIOPropertySCSIInitiatorIdentifierKey
@discussion An identifier that will uniquely identify this SCSI Initiator for the
SCSI Domain.

Requirement: Mandatory for SCSI Parallel Interface, SAS,
and Fibre Channel Interface.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
		<key>SCSI Initiator Identifier</key>
		<integer>7</integer>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySCSIInitiatorIdentifierKey		"SCSI Initiator Identifier"


/*!
@defined kIOPropertySCSIDomainIdentifierKey
@discussion An identifier that will uniquely identify this SCSI Domain for the
Physical Interconnect type. This identifier is only guaranteed to be unique for
any given Physical Interconnect and is not guaranteed to be the same across
restarts or shutdowns.

Requirement: Mandatory for SCSI Parallel Interface and Fibre Channel Interface.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
		<key>SCSI Domain Identifier</key>
		<integer>0</integer>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySCSIDomainIdentifierKey			"SCSI Domain Identifier"


/*!
@defined kIOPropertySCSITargetIdentifierKey
@discussion This is the SCSI Target Identifier for a given SCSI Target Device.

Requirement: Mandatory for SCSI Parallel Interface and Fibre Channel Interface.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
		<key>SCSI Target Identifier</key>
		<integer>3</integer>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySCSITargetIdentifierKey			"SCSI Target Identifier"


/*!
@defined kIOPropertySCSILogicalUnitNumberKey
@discussion This key is the SCSI Logical Unit Number for the device server
controlled by the driver.

Requirement: Mandatory for SCSI Parallel Interface, SAS, and Fibre Channel Interface.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
		<key>SCSI Logical Unit Number</key>
		<integer>2</integer>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySCSILogicalUnitNumberKey			"SCSI Logical Unit Number"


/*!
@defined kIOPropertyPhysicalInterconnectTypeKey
@discussion This key is used to define the Physical Interconnect to which
a device is attached.

Requirement: Mandatory.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeKey		"Physical Interconnect"


/*!
@defined kIOPropertyPhysicalInterconnectLocationKey
@discussion This key is used to define the Physical Interconnect
Location.

Requirement: Mandatory.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectLocationKey	"Physical Interconnect Location"


/*!
@defined kIOPropertySCSIProtocolMultiInitKey
@discussion This protocol characteristics key is used to inform the system
that the protocol supports having multiple devices that act as initiators.

Requirement: Optional.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>Fibre Channel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
		<key>Multiple Initiators</key>
		<string>True</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySCSIProtocolMultiInitKey			"Multiple Initiators"


/*
 *	Values - Values for the characteristics defined above.
 */


/*!
@defined kIOPropertyInternalKey
@discussion This key defines the value of Internal for the key
kIOPropertyPhysicalInterconnectLocationKey. If the device is
connected to an internal bus, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>ATA</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyInternalKey						"Internal"


/*!
@defined kIOPropertyExternalKey
@discussion This key defines the value of External for the key
kIOPropertyPhysicalInterconnectLocationKey. If the device is
connected to an external bus, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>Fibre Channel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyExternalKey						"External"


/*!
@defined kIOPropertyInternalExternalKey
@discussion This key defines the value of Internal/External for the key
kIOPropertyPhysicalInterconnectLocationKey. If the device is connected
to a bus and it is indeterminate whether it is internal or external,
this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>Internal/External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyInternalExternalKey				"Internal/External"


/*!
@defined kIOPropertyInterconnectFileKey
@discussion This key defines the value of File for the key
kIOPropertyPhysicalInterconnectLocationKey. If the device is a file
that is being represented as a storage device, this key should be set.

NOTE: This key should only be used when the Physical Interconnect is set to
Virtual Interface.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>Virtual Interface</string>
		<key>Physical Interconnect Location</key>
		<string>File</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyInterconnectFileKey						"File"


/*!
@defined kIOPropertyInterconnectRAMKey
@discussion This key defines the value of RAM for the key
kIOPropertyPhysicalInterconnectLocationKey. If the device is system memory
that is being represented as a storage device, this key should be set.

NOTE: This key should only be used when the Physical Interconnect is set to
Virtual Interface.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>Virtual Interface</string>
		<key>Physical Interconnect Location</key>
		<string>RAM</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyInterconnectRAMKey						"RAM"


/*!
@defined kIOPropertyPhysicalInterconnectTypeATA
@discussion This key defines the value of ATA for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to an ATA bus, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>ATA</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeATA				"ATA"


/*!
@defined kIOPropertyPhysicalInterconnectTypeSerialATA
@discussion This key defines the value of SATA for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to a SATA bus, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SATA</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeSerialATA		"SATA"


/*!
@defined kIOPropertyPhysicalInterconnectTypeSerialAttachedSCSI
@discussion This key defines the value of SAS for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to a SAS bus, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SAS</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeSerialAttachedSCSI	"SAS"


/*!
@defined kIOPropertyPhysicalInterconnectTypeATAPI
@discussion This key defines the value of ATAPI for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to an ATA bus and follows the ATAPI command specification, this key
should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>ATAPI</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeATAPI			"ATAPI"


/*!
@defined kIOPropertyPhysicalInterconnectTypeUSB
@discussion This key defines the value of USB for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to a USB port, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>USB</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeUSB				"USB"


/*!
@defined kIOPropertyPhysicalInterconnectTypeFireWire
@discussion This key defines the value of USB for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to a FireWire port, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>FireWire</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeFireWire			"FireWire"


/*!
 @defined kIOPropertyPhysicalInterconnectTypeSecureDigital
 @discussion This key defines the value of Secure Digital for the key
 kIOPropertyPhysicalInterconnectTypeSecureDigital. If the device is
 connected to a Secure Digital port and follows the Secure Digital 
 specification, this key should be set. 
 
 Example:
 <pre>
 @textblock
 <dict>
    <key>Protocol Characteristics</key>
    <dict>
        <key>Physical Interconnect</key>
        <string>Secure Digital</string>
        <key>Physical Interconnect Location</key>
        <string>Internal</string>
    </dict>
 </dict>
 @/textblock
 </pre>
 */
#define kIOPropertyPhysicalInterconnectTypeSecureDigital	"Secure Digital"


/*!
@defined kIOPropertyPhysicalInterconnectTypeSCSIParallel
@discussion This key defines the value of SCSI Parallel Interface for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to a SCSI Parallel port, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>SCSI Parallel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeSCSIParallel		"SCSI Parallel Interface"


/*!
@defined kIOPropertyPhysicalInterconnectTypeFibreChannel
@discussion This key defines the value of Fibre Channel Interface for the key
kIOPropertyPhysicalInterconnectTypeKey. If the device is connected
to a Fibre Channel port, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>Fibre Channel Interface</string>
		<key>Physical Interconnect Location</key>
		<string>External</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeFibreChannel		"Fibre Channel Interface"


/*!
@defined kIOPropertyPhysicalInterconnectTypeVirtual
@discussion This key defines the value of Virtual Interface for the key
kIOPropertyPhysicalInterconnectTypeVirtual. If the device is being made to look
like a storage device, but is not such in actuality, such as a File or RAM, this
key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>Virtual Interface</string>
		<key>Physical Interconnect Location</key>
		<string>File</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypeVirtual		"Virtual Interface"


/*!
@defined kIOPropertyPhysicalInterconnectTypePCI
@discussion This key defines the value of PCI for the key
kIOPropertyPhysicalInterconnectTypePCI. If the device is connected
via PCI, this key should be set.

Example:
<pre>
@textblock
<dict>
	<key>Protocol Characteristics</key>
	<dict>
		<key>Physical Interconnect</key>
		<string>PCI</string>
		<key>Physical Interconnect Location</key>
		<string>Internal</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyPhysicalInterconnectTypePCI		"PCI"

/*!
 @defined kIOPropertyPhysicalInterconnectTypePCIExpress
 @discussion This key defines the value of PCI-Express for the key
 kIOPropertyPhysicalInterconnectTypePCIExpress. If the device is connected
 via PCI-Express, this key should be set.
 
 Example:
 <pre>
 @textblock
 <dict>
 	<key>Protocol Characteristics</key>
 	<dict>
 		<key>Physical Interconnect</key>
 		<string>PCI-Express</string>
 		<key>Physical Interconnect Location</key>
 		<string>Internal</string>
 	</dict>
 </dict>
 @/textblock
 </pre>
 */
#define kIOPropertyPhysicalInterconnectTypePCIExpress		"PCI-Express"

/*!
 @defined kIOPropertyPhysicalInterconnectTypeAppleFabric
 @discussion This key defines the value of Apple Fabric for the key
 kIOPropertyPhysicalInterconnectTypeAppleFabric. If the device is connected
 via Apple Fabric, this key should be set.

 Example:
 <pre>
 @textblock
 <dict>
 	<key>Protocol Characteristics</key>
 	<dict>
 		<key>Physical Interconnect</key>
 		<string>Apple Fabric</string>
 		<key>Physical Interconnect Location</key>
 		<string>Internal</string>
 	</dict>
 </dict>
 @/textblock
 </pre>
 */
#define kIOPropertyPhysicalInterconnectTypeAppleFabric		"Apple Fabric"

#endif	/* _IOKIT_IO_STORAGE_PROTOCOL_CHARACTERISTICS_H_ */
