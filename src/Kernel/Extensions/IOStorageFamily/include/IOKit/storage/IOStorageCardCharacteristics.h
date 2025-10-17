/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#ifndef _IOKIT_IO_STORAGE_CARD_CHARACTERISTICS_H_
#define _IOKIT_IO_STORAGE_CARD_CHARACTERISTICS_H_

		
/*
 *	Card Characteristics - Characteristics defined for cards.
 */

/*!
@defined kIOPropertyCardCharacteristicsKey
@discussion This key is used to define Card Characteristics for a particular
piece of MMC/SD media and it has an associated dictionary which lists the
card characteristics.
 
Requirement: Mandatory

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardCharacteristicsKey				"Card Characteristics"


/*!
@defined kIOPropertySlotKey
@discussion This key is used to define the slot number for the device
 
Requirement: Mandatory

Example:
<pre>
@textblock
<dict>
	<key>Slot</key>
	<integer>1<integer>
 </dict>
@/textblock
</pre>
*/
#define kIOPropertySlotKey								"Slot"


/*!
@defined kIOProperty64BitKey
@discussion This key defines wether the device supports 64-bit.
 
Requirement: Mandatory

Example:
<pre>
@textblock
<dict>
	<key>64-bit</key>
	<true/>
</dict>
@/textblock
</pre>
*/
#define kIOProperty64BitKey								"64-bit"


/*!
@defined kIOPropertyClockDivisorKey
 @discussion This key defines the current clock divisor for the device.

Requirement: Mandatory.

Example:
<pre>
@textblock
<dict>
	<key>Clock Divisor</key>
	<integer>128</integer>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyClockDivisorKey						"Clock Divisor"


/*!
@defined kIOPropertyBaseFrequencyKey
@discussion This key defines the current base frequency for the device.
 
Requirement: Mandatory.

Example:
<pre>
@textblock
<dict>
	<key>Base Frequency</key>
	<integer>50</integer>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyBaseFrequencyKey						"Base Frequency"


/*!
@defined kIOPropertyBusVoltageKey
@discussion This key defines the current bus voltage for the device in mV
 
Requirement: Mandatory.

Example:
<pre>
@textblock
<dict>
	<key>Bus Voltage</key>
	<integer>3300</integer>
</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyBusVoltageKey						"Bus Voltage"


/*!
@defined kIOPropertyBusWidthKey
@discussion This key defines the current bus width for the device.

Requirement: Mandatory.
 
Example:
<pre>
@textblock
<dict>
	<key>Bus Width</key>
	<integer>4</integer>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyBusWidthKey							"Bus Width"


/*!
@defined kIOPropertyCardPresentKey
@discussion This key defines wether a MMC or SD card is physically present.
 
Requirement: Mandatory

Example:
<pre>
@textblock
<dict>
	<key>Card Present</key>
	<true/>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardPresentKey						"Card Present"


/*!
 @defined kIOPropertyProductSerialNumberKey
 @discussion This key is used to indicate the card serial number ID.
 
 Requirement: Mandatory
 
 Example:
 <pre>
 @textblock
 <dict>
	 <key>Card Characteristics</key>
	 <dict>
		 <key>Product Name</key>
		 <string>SD32G</string>
		 <key>Product Revision Level</key>
		 <string>1.0</string>
		 <key>Card Type</key>
		 <string>SDHC</string>
		 <key>Serial Number</key>
		 <data>0045ff</data>
	 </dict>
 </dict>
 @/textblock
 </pre>
 */
#define kIOPropertyProductSerialNumberKey				"Serial Number"


/*!
 @defined kIOPropertyManufacturerIDKey
 @discussion This key is used to indicate the card manufacturer ID.
 
 Requirement: Optional
 
 Example:
 <pre>
 @textblock
 <dict>
	 <key>Card Characteristics</key>
	 <dict>
		 <key>Product Name</key>
		 <string>SD32G</string>
		 <key>Product Revision Level</key>
		 <string>1.0</string>
		<key>Card Type</key>
		<string>SDHC</string>
		<key>Manufacturer ID</key>
		<data>03</data>
	 </dict>
 </dict>
 @/textblock
 </pre>
 */
#define kIOPropertyManufacturerIDKey					"Manufacturer ID"


/*!
@defined kIOPropertyApplicationIDKey
 @discussion This key is used to indicate the card application ID.

Requirement: Optional

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>SDHC</string>
		<key>Application ID</key>
		<data>ffff</data>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyApplicationIDKey						"Application ID"


/*!
@defined kIOPropertyManufacturingDateKey
 @discussion This key is used to indicate the card manufacturing date.

Requirement: Mandatory.
 
Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>SDHC</string>
		<key>Manufacturing Date</key>
		<string>2009-12</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyManufacturingDateKey					"Manufacturing Date"


/*!
@defined kIOPropertySpeedClassKey
 @discussion This key is used to indicate SD card speed class.

Requirement: Mandatory.
 
Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>SDHC</string>
		<key>Speed Class</key>
		<data>02</data>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySpeedClassKey						"Speed Class"


/*!
@defined kIOPropertySpecificationVersionKey
@discussion This key is used to indicate the card specification version.

Requirement: Mandatory.

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>SDHC</string>
		<key>Specification Version</key>
		<string>3.0</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertySpecificationVersionKey				"Specification Version"


/*!
@defined kIOPropertyCardTypeKey
 @discussion This key is used to indicate the card type is MMC.

Requirement: Optional.

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>MMC</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardTypeKey							"Card Type"


/*!
@defined kIOPropertyCardTypeMMCKey
 @discussion This key is used to indicate the card type is MMC.

 Requirement: Optional.

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		 <key>Product Name</key>
		 <string>SD32G</string>
		 <key>Product Revision Level</key>
		 <string>1.0</string>
		 <key>Card Type</key>
		 <string>MMC</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardTypeMMCKey						"MMC"


/*!
@defined kIOPropertyCardTypeSDSCKey
 @discussion This key is used to indicate the card type is SDSC.

Requirement: Optional.

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		 <key>Product Name</key>
		 <string>SD32G</string>
		 <key>Product Revision Level</key>
		 <string>1.0</string>
		 <key>Card Type</key>
		 <string>SDSC</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardTypeSDSCKey						"SDSC"


/*!
@defined kIOPropertyCardTypeSDHCKey
 @discussion This key is used to indicate the card type is SDHC.

Requirement: Optional.

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>SDHC</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardTypeSDHCKey						"SDHC"


/*!
@defined kIOPropertyCardTypeSDXCKey
 @discussion This key is used to indicate the card type is SDXC.

Requirement: Optional.

Example:
<pre>
@textblock
<dict>
	<key>Card Characteristics</key>
	<dict>
		<key>Product Name</key>
		<string>SD32G</string>
		<key>Product Revision Level</key>
		<string>1.0</string>
		<key>Card Type</key>
		<string>SDXC</string>
	</dict>
</dict>
@/textblock
</pre>
*/
#define kIOPropertyCardTypeSDXCKey						"SDXC"


#endif	/* _IOKIT_IO_STORAGE_CARD_CHARACTERISTICS_H_ */
